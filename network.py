import torch

from mlp import MLPBase
import torch.nn as nn
from easydict import EasyDict

import sys
from pathlib import Path
import os

sys.path.append(str(Path(os.path.dirname(__file__)).parent.resolve()))
# from envs.train_wrapper import TrainWrapper
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
import numpy as np
from envs.feature import FeatureParser

from mask import MaskedPolicy

cfg = CompetitionConfig()
env = TeamBasedEnv(config=cfg)
feature_parser = FeatureParser()

args = {
    'use_feature_normalization': 1,
    'use_orthogonal': 1,
    'use_ReLU': 1,
    'stacked_frames': 0,
    'layer_N': 2,
    'hidden_size': 128
}

args = EasyDict(args)





class NMMO(nn.Module):
    def __init__(self, args):
        super(NMMO, self).__init__()
        self.args = args


        # 卷积网络
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=17,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU())


        self.policy1 = nn.Linear(128, 5)
        self.policy2 = nn.Linear(128, 61)
        self.baseline = nn.Linear(128, 1)


    def forward(self, obs_dict, training=False):
        # 处理vector
        obs_tensor = {}
        for vector_name, vector in obs_dict.items():
            obs_tensor[vector_name] = torch.from_numpy(vector)
        for vector_name, vector in obs_tensor.items():
            vector_mlp = MLPBase(args=self.args, obs_shape=vector.shape)
            vector_out = vector_mlp(vector)
            obs_tensor[vector_name] = vector_out.view(8, -1)

        # 处理local_map
        local_map = torch.from_numpy(obs_dict['local_map'])
        raw_out = self.cnn(local_map).view(8, -1)
        map_mlp = MLPBase(args=args, obs_shape=raw_out.shape)
        local_map_out = map_mlp(raw_out)
        obs_tensor.update(local_map_out=local_map_out)

        trans_matrix = torch.stack([vector for vector in obs_tensor.values()], dim=1).view(8, -1)

        move_logits = self.policy1(trans_matrix)
        attack_logits = self.policy2(trans_matrix)
        baseline = self.baseline(trans_matrix)

        move_va = obs_dict.get('move_va', None)
        attack_va = obs_dict.get('attack_va', None)

        move_dist = MaskedPolicy(move_logits, valid_actions=move_va)
        attack_dist = MaskedPolicy(attack_logits, valid_actions=attack_va)
        if not training:
            move_action = move_dist.sample()
            attack_action = attack_dist.sample()
        else:
            move_action = None
            attack_action = None

        policy_move_logits = move_dist.logits.view(8, -1)
        policy_attack_logits = attack_dist.logits.view(8, -1)
        baseline = baseline

        output = dict(
            policy_move_logits=policy_move_logits,
            policy_attack_logits=policy_attack_logits,
            baseline=baseline
        )

        if move_action is not None:
            output.update(move_action=move_action)

        if attack_action is not None:
            output.update(attack_action=attack_action)

        return output


while True:
    obs = env.reset()
    obs_dict = feature_parser.parse(obs[0], cfg)
    agent_vector = obs_dict['agent_vector']
    agent_vector = torch.from_numpy(agent_vector)
    npc_vector = torch.from_numpy(obs_dict['npc_vector'])

    mlp = MLPBase(
        args=args,
        obs_shape=[agent_vector.shape[1]],
    )

    out = mlp(agent_vector)
    print(out.shape)
