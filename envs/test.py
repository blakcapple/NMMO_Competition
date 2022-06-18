"""
用于测试与环境有关的函数
"""
import sys 
from pathlib import Path
import os
sys.path.append(str(Path(os.path.dirname(__file__)).parent.resolve()))
from envs.train_wrapper import TrainWrapper
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
import numpy as np 
from envs.feature import FeatureParser

cfg = CompetitionConfig()
env = TeamBasedEnv(config=cfg)
feature_parser = FeatureParser()
while True:
    obs = env.reset()
    obs_dict = feature_parser.parse(obs[0])
    actions = np.zeros((8,1))
    obs_array, share_obs_array, reward_array, done_array, info, available_actions  = env.step(actions)