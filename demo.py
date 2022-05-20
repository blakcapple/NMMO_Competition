# import numpy as np 
# def onehot_initialization(a):
#     def all_idx(idx, axis):
#         grid = np.ogrid[tuple(map(slice, idx.shape))]
#         grid.insert(axis, idx)
#         return tuple(grid)
#     ncols = a.max() + 1
#     out = np.zeros(a.shape + (ncols,), dtype=int)
#     out[all_idx(a, axis=2)] = 1 
#     return out 

# def all_idx(idx, axis):
#     grid = np.ogrid[tuple(map(slice, idx.shape))]
#     grid.insert(axis, idx)
#     return tuple(grid)

# a = np.random.randint(0,9, (10,10))
# b = onehot_initialization(a)
# print(b)
from envs.train_wrapper import TrainWrapper
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
import numpy as np 
def create_env():
    cfg = CompetitionConfig()
    return (TrainWrapper(TeamBasedEnv(config=cfg)))
import time 
start_time = time.time()
env = create_env()
env.reset()
step = 0
while True :
    actions = np.zeros((8,1))
    # env_actions = {agent_id: actions[agent_id] for agent_id in range(actions.shape[0])}
    obs_array, share_obs_array, reward_array, done_array, info, available_actions  = env.step(actions)
    if np.all(done_array):
        env.reset()
    step += 1
    if step % 10 == 0:
        print(time.time() - start_time)
        start_time = time.time()




