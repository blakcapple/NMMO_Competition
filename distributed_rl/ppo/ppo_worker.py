from distutils.log import error
import ray 
from distributed_rl.common.worker import RLWorker
import torch.nn as nn 
import numpy as np 
import torch 
from distributed_rl.ppo.utils import LocalBuffer, set_seed, create_env
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
import os 
from tensorboardX import SummaryWriter
from copy import deepcopy
import wandb
import socket

@ray.remote
class PPOWorker(RLWorker):
    def __init__(
        self, worker_id: int, config: dict, port_cfg:dict,):
        
        super().__init__(worker_id, port_cfg)

        all_args = config['all_args']
        self.device = config["worker_device"]
        self.worker_buffer_size = all_args.episode_length
        self.run_dir = config['run_dir']
        self.num_agents = config['num_agents']
        self.model_dir = config['model_dir']
        self.use_wandb = all_args.use_wandb
        self.recurrent_N = all_args.recurrent_N
        self.hidden_size = all_args.hidden_size
        self.use_centralized_V = all_args.use_centralized_V
        # create env
        self.env = create_env(all_args.team_spirit, all_args.action_type)
        self.seed = worker_id*100 + all_args.seed
        self.env.seed(self.seed)
        set_seed(self.seed)

        self.buffer = LocalBuffer(self.worker_buffer_size, 
                                  self.num_agents,
                                  self.env.action_space,
                                 ) # local buffer
        self.policy = Policy(all_args,
                            self.env.action_space,
                            device = self.device)

    def warmup(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        obs, available_actions, info = self.env.reset()
        self.obs = obs
        self.available_actions = available_actions
        if isinstance(obs, dict):
            for key_1 in obs.keys():
                for key_2 in obs[key_1].keys():
                    self.buffer.obs[key_1][key_2][0] = obs[key_1][key_2].copy()
        else:
            self.buffer.obs[0] = obs.copy()
        if isinstance(available_actions, list):
            for index, action in enumerate(available_actions):
                self.buffer.available_actions[index][0] = action.copy()
        else:
            self.buffer.available_actions[0] = available_actions.copy()
        
        # self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        if isinstance(self.obs, dict):
            for key, value in self.obs.items():
                for key_1, value_1 in value.items():
                    self.obs[key][key_1] = value_1[np.newaxis,:] # 插入维度1方便后面的网络计算
        values, actions, action_log_probs, rnn_states, rnn_states_critic \
         = self.policy.get_actions(
                                    self.obs,
                                    self.buffer.rnn_states[step],
                                    self.buffer.rnn_states_critic[step],
                                    self.buffer.masks[step],
                                    self.available_actions)
        return values.numpy(), actions.numpy(), action_log_probs.numpy(), rnn_states.numpy(),\
               rnn_states_critic.numpy(), 

    def collect_data(self):

        for step in range(self.buffer.size):
            
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

            obs, rewards, dones, infos, available_actions = self.env.step(actions)

            self.obs = deepcopy(obs) 
            self.available_actions = deepcopy(available_actions)

            data = obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
            
            self.insert(data)
    
    def insert(self, data):
        obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones)

        if dones_env:

            rnn_states = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((self.num_agents, *self.buffer.rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.num_agents, 1), dtype=np.float32)
        if dones_env == True:
            masks = np.zeros((self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = 0
        if dones_env:
            active_masks = np.ones((self.num_agents, 1), dtype=np.float32)
        
        self.buffer.insert(obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, active_masks, available_actions)

    def act(self):

        self.collect_data() # 收集数据 直到local buffer 满
        local_buffer = self.buffer.get()
        self.send_replay_data(local_buffer) # 向global buffer 发送 local buffer 的数据
        self.receive_new_params() # 接收新的model参数
        print(f'work_{self.worker_id} has received params from learner')
        self.buffer.after_update()
        # self.receive_new_learning_stage()

    def run(self):
        try:
            print(f'worker {self.worker_id} starts running')
            self.receive_new_params()
            print(f'work_{self.worker_id} has received params from learner')
            if self.worker_id == 1:
                """
                evaluate
                """
                self.test_run()
            else:
                self.warmup()
                while True:
                    self.act()
        except KeyboardInterrupt:
            import sys
            sys.exit()

    def test_run(self):
        episode_reward = []
        episode_step = []
        average_reward_sequence = []
        agent_reward_sequence = []
        agent_episode_step_sequence = []
        average_episode_step_sequence = []
        episode = 0
        eval_obs, eval_available_actions, info = self.env.reset()
        self.policy.actor.eval()
        self.policy.critic.eval()
        best_reward = -np.inf 
        while True:
            eval_rnn_states = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.num_agents, 1), dtype=np.float32)
            if isinstance(eval_obs, dict):
                for key, value in eval_obs.items():
                    for key_1, value_1 in value.items():
                        eval_obs[key][key_1] = value_1[np.newaxis,:] # 插入维度1方便后面的网络计算
            eval_actions, eval_rnn_states = \
            self.policy.act(eval_obs,
                            eval_rnn_states,eval_masks, 
                            eval_available_actions, 
                            deterministic=True)
            eval_actions = eval_actions.numpy()
            eval_rnn_states = eval_rnn_states.numpy()
            eval_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.env.step(eval_actions)
            eval_dones_env = np.all(eval_dones)
            if eval_dones_env:
                eval_rnn_states = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.num_agents, 1), dtype=np.float32)
            if eval_dones_env:
                eval_masks = np.zeros((self.num_agents, 1), dtype=np.float32)
            episode_reward.append(eval_rewards) 
            episode_step.append(~np.array(eval_dones, dtype=bool))
            if eval_dones_env:
                agent_reward = np.sum(episode_reward, axis=0)
                agent_episode_step = np.sum(episode_step, axis=0)
                average_reward = np.mean(agent_reward)
                average_step = np.mean(agent_episode_step)
                agent_reward_sequence.append(agent_reward)
                agent_episode_step_sequence.append(agent_episode_step)
                average_reward_sequence.append(average_reward)
                average_episode_step_sequence.append(average_step)
                # stage = eval_infos
                episode += 1
                evaluate_dict = dict(average_reward=np.mean(average_reward_sequence[-20:]),
                                        average_episode_step=np.mean(average_episode_step_sequence[-20:]),
                                        )
                for i in range(8):
                    evaluate_dict[f'agent_{i}_reward'] = np.mean(np.array(agent_reward_sequence)[:,i][-20:])
                    evaluate_dict[f'agent_{i}_episode_step'] = np.mean(np.array(agent_episode_step_sequence)[:,i][-20:])
                for key, value in eval_infos.items():
                    evaluate_dict[key] = value
                self.send_evaluate_data(evaluate_dict)
                if best_reward < average_reward:
                    best_reward = average_reward
                print(f'Reward:{average_reward:.3f}, Best Reward:{best_reward:.3f}')
                self.receive_new_params(wait=True)
                print(f'work_{self.worker_id} has received params from learner')
                episode_reward = []
                episode_step = []