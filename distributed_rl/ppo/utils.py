from collections import namedtuple
import enum
import random
import torch 
import numpy as np 
from envs.train_wrapper import TrainWrapper
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'active_mask',
                                       'reward', 'value', 'log_prob', 'share_obs',
                                       'available_action', 'rnn_states', 'rnn_states_critic'))

def create_env(team_sprit=0):
    cfg = CompetitionConfig()
    cfg.NMAPS = 400
    return TrainWrapper(TeamBasedEnv(config=cfg), team_sprit=team_sprit)

def _t2n(x):
    return x.detach().cpu().numpy()

def set_seed(seed):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


class LocalBuffer:

    def __init__(self, size, num_agents, act_space, hidden_size=64, recurrent_N=1):

        self.size = size
        self.multi_discrete = False
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.size + 1, num_agents, act_space.n),
                                             dtype=np.float32)
        elif act_space.__class__.__name__ == 'MultiDiscrete':
            self.multi_discrete = True
            action_dims = act_space.nvec
            self.available_actions = []
            for action_dim in action_dims:
                available_action = np.ones((self.size + 1, num_agents, action_dim),
                                             dtype=np.float32)
                self.available_actions.append(available_action)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        # obs_shape = get_shape_from_obs_space(obs_space)
        # share_obs_shape = get_shape_from_obs_space(share_obs_space)


        # self.obs = np.zeros((size + 1, num_agents, *obs_shape), dtype=np.float32)
        local_obs = {
            "agent_vector": np.zeros((size+1, num_agents, 17),dtype=np.float32),
            "local_map": np.zeros((size+1, num_agents, 17, 15, 15), dtype=np.float32),
            "attack_vector": np.zeros((size+1, num_agents, 20*12), dtype=np.float32),
                    } 
        global_obs = {
            "npc_vector": np.zeros((size+1, 20, 17), dtype=np.float32),
            "enemy_vector": np.zeros((size+1, 20, 17), dtype=np.float32),
            "team_vector": np.zeros((size+1, num_agents*17), dtype=np.float32),
            "time": np.zeros((size+1, 1), dtype=np.float32),
                    }
        self.obs = {'local_obs':local_obs, "global_obs":global_obs
                    }

        self.rnn_states = np.zeros(
            (size + 1, num_agents, recurrent_N, hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (size + 1, num_agents, 1), dtype=np.float32)

        self.returns = np.zeros_like(self.value_preds)
        if self.multi_discrete:
            self.actions = np.zeros(
                (size, num_agents, *act_shape), dtype=np.float32)
            self.action_log_probs = np.zeros(
                (size, num_agents, *act_shape), dtype=np.float32)
        else:
            self.actions = np.zeros(
                (size, num_agents, act_shape), dtype=np.float32)
            self.action_log_probs = np.zeros(
                (size, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (size, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((size + 1, num_agents, 1), dtype=np.float32)

        self.active_masks = np.ones_like(self.masks)


        self.step = 0

    def insert(self, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, active_masks=None, available_actions=None):

        for key_1 in obs.keys():
            for key_2 in obs[key_1].keys():
                self.obs[key_1][key_2][self.step + 1] = obs[key_1][key_2].copy()

        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            if self.multi_discrete:
                for index, action in enumerate(available_actions):
                    self.available_actions[index][self.step + 1] = action.copy()
            else:
                self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.size

    def compute_return(self, next_value, value_normalizer=None):

            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step] 

    def get(self):

        """
        return all data in the buffer
        """    
        return dict(obs=self.obs,
                    rnn_states=self.rnn_states,
                    rnn_states_critic=self.rnn_states_critic,
                    available_actions=self.available_actions,
                    masks=self.masks,
                    value_preds=self.value_preds,
                    actions=self.actions,
                    action_log_probs=self.action_log_probs,
                    rewards=self.rewards,
                    active_masks=self.active_masks,)

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""

        # self.obs[0] = self.obs[-1].copy()
        for key_1 in self.obs.keys():
            for key_2 in self.obs[key_1].keys():
                self.obs[key_1][key_2][0] = self.obs[key_1][key_2][-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            if self.multi_discrete:
               for avaliable_action in self.available_actions:
                    avaliable_action[0] = avaliable_action[-1].copy()
            else:
                self.available_actions[0] = self.available_actions[-1].copy()
