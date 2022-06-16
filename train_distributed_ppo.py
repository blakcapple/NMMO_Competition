from email.policy import default
import os
import sys
import setproctitle
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from pathlib import Path
import torch
from config import get_config
from distributed_rl.ppo.runner import DistributedPPO as Runner
import wandb 
import socket
import numpy as np 
from distributed_rl.ppo.ppo_worker import PPOWorker
from distributed_rl.ppo.ppo_learner import PPOLearner
import json 

def parse_args(args, parser):
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--skill', default='Exploration', type=str)
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # for debug:
    # all_args.n_rollout_threads = 1
    # all_args.use_wandb = False 
    # all_args.use_recurrent_policy = False 

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError
   
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        learner_device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        learner_device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path((os.path.dirname(os.path.abspath(__file__)))
                        + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    model_dir = Path(os.path.dirname(__file__)).resolve() / 'load_model'
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))    
    
    # 设置进程的别名
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    config = {
        "all_args": all_args,
        "num_agents": 8,
        "worker_device": torch.device("cpu"),
        "learner_device": learner_device,
        "run_dir": run_dir,
        'num_workers': all_args.n_rollout_threads+1,
        'batch_size': all_args.n_rollout_threads*all_args.episode_length,
        'num_learners': 1,
        'model_dir':model_dir
    }
    with open(str(run_dir)+'/arguments.txt', 'w') as f:
        json.dump(all_args.__dict__, f, indent=2)
    port_cfg = dict(pubsub_port=4111,pullpush_port=4112, pair_port=4113, pubsub_port2=4114,)
    runner = Runner(PPOWorker, PPOLearner, port_cfg, config)
    runner.spawn()
    runner.train()

if __name__ == "__main__":
    main(sys.argv[1:])