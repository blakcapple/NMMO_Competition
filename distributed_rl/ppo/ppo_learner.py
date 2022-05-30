
from distributed_rl.common.learner import Learner
import ray 
import pyarrow as pa
import numpy as np  
from copy import deepcopy
from algorithms.r_mappo.r_mappo import R_MAPPO as Trainer
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from tensorboardX import SummaryWriter
import os 
from distributed_rl.ppo.shared_buffer import SharedReplayBuffer
import sys 
import zmq 
import wandb 
from distributed_rl.ppo.utils import create_env, set_seed
import torch 
from distributed_rl.ppo.utils import _t2n
import time 

@ray.remote(num_gpus=1)
class PPOLearner(Learner):

    def __init__(self, port_cfg, config):

        super().__init__(port_cfg)

        all_args = config['all_args']
        set_seed(all_args.seed)
        self.device = config['learner_device']
        self.num_agents = config['num_agents']
        self.batch_size = config['batch_size']
        self.run_dir = config['run_dir']
        self.model_dir = config['model_dir']
        self.load = all_args.load
        self.num_env_steps = all_args.num_env_steps
        env = create_env()
        share_observation_space = env.share_observation_space if all_args.use_centralized_V else env.observation_space
        self.policy = Policy(all_args, 
                             env.observation_space,
                             share_observation_space,
                             env.action_space, 
                             self.device)
        self.trainer = Trainer(all_args, self.policy, self.device) 
        self.buffer = SharedReplayBuffer(all_args, 
                                        self.num_agents, 
                                        env.observation_space,
                                        env.share_observation_space,
                                        env.action_space) 
        self.use_wandb = all_args.use_wandb
        self.workers_num = all_args.n_rollout_threads # the num of sampler worker
        self.n_rollout_threads = all_args.n_rollout_threads 
        self.samples_num = 0
        self.save_interval = all_args.save_interval
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        del env 
        if self.load:
            self.restore()
            print('load model!')

    def recv_replay_data_(self):
    
        new_replay_data_id = self.pull_socket.recv()
        replay_data = pa.deserialize(new_replay_data_id)
        return replay_data

    def publish_params(self, new_params: np.ndarray):
    
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_evaluate_data(self):
        new_evaluate_data_id = False
        try: 
            new_evaluate_data_id = self.pair_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if new_evaluate_data_id:
            new_data = pa.deserialize(new_evaluate_data_id)
            self.evaluate_dict = new_data

    def get_params(self):

        actor_params = []
        critic_params = []
        actor = deepcopy(self.trainer.policy.actor)
        actor_state_dict = actor.cpu().state_dict()
        for param in list(actor_state_dict):
            actor_params.append(actor_state_dict[param].numpy())
        
        critic = deepcopy(self.trainer.policy.critic)
        critic_state_dict = critic.cpu().state_dict()
        for param in list(critic_state_dict):
            critic_params.append(critic_state_dict[param].numpy())

        return (actor_params, critic_params)

    def build_buffer(self):
        data_key = ['obs', 'share_obs', 'actions', 'available_actions', 'rnn_states', 
                    'rnn_states_critic', 'rewards', 'masks', 'active_masks']
        all_data = dict()
        for key in data_key:
            all_data[key] = []
        for _ in range(self.workers_num):
            replay_data = self.recv_replay_data_()
            for key in data_key:
                all_data[key].append(replay_data[key])
        for key in data_key:
            assert hasattr(self.buffer, key), 'check keys!'
            setattr(self.buffer, key, np.stack(all_data[key], axis=1))
        

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        return train_infos

    def run(self):

        try:
            start_time = time.time()
            self.best_reward = -np.inf
            self.update_step = 0
            self.evaluate_dict = {}
            log_dir = os.path.join(self.run_dir, 'learner')
            self.logger = SummaryWriter(log_dir)
            # 同步所有的worker的参数
            params = self.get_params()
            self.publish_params(params)
            while self.samples_num <= self.num_env_steps:
                self.build_buffer()
                self.compute()
                train_infos = self.train()
                params = self.get_params()
                self.publish_params(params)
                self.recv_evaluate_data()
                self.update_step += 1
                self.samples_num = self.batch_size * self.update_step
                time_used = time.time() - start_time
                print(f'FPS:{self.samples_num / time_used:.2f}, best_reward:{self.best_reward}, update_step:{self.update_step}')
                if self.evaluate_dict:
                    all_log_info = {**self.evaluate_dict, **train_infos}
                    if all_log_info['average_reward'] > self.best_reward:
                        self.best_reward = all_log_info['average_reward']
                        self.save(self.update_step, True)
                    self.log_info(all_log_info)
                    self.evaluate_dict = {}
                if self.update_step % self.save_interval == 0:
                    self.save(self.update_step)

        except KeyboardInterrupt:
            sys.exit()

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir)+'/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, data:dict):
        if self.use_wandb:
            for key, value in data.items():
                wandb.log({key:value}, step=self.samples_num)
        else: 
            for key, value in data.items():
                self.logger.add_scalar(key, value, self.samples_num)

    def save(self, epoch, save_best=False):
        """Save policy's actor and critic networks."""
        if save_best:
            # remove last best model
            files = os.listdir(self.save_dir)
            for file in files:
                if 'best' in file:
                    file_path = os.path.join(self.save_dir, file)
                    os.remove(file_path)
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{epoch}_best.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{epoch}_best.pt")
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{epoch}.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{epoch}.pt")