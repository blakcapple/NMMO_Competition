from distributed_rl.common.architecture import Architecture
import ray 
import sys 

class DistributedPPO(Architecture):
    
    def __init__(self, worker_cls:type,
                       learner_cls:type,
                       port_cfg:dict,
                       config:dict,):
        super().__init__(worker_cls, learner_cls, config)

        all_args = config['all_args']
        self.num_agents = config['num_agents']
        self.config = config
        self.port_config = port_cfg
        self.use_centralized_V = all_args.use_centralized_V
    
    def spawn(self):

        self.workers = [
            self.worker_cls.remote(n, self.config, self.port_config)
            for n in range(1, self.num_workers+1)
        ]
        self.learner = self.learner_cls.remote(self.port_config, self.config)
        self.all_actors = self.workers + [self.learner]

    def train(self):
        learner_id = self.learner.run.remote()
        worker_1_id = self.workers[0].run.remote()
        id = ray.wait([actor.run.remote() for actor in self.all_actors])
        if id == learner_id:
            print('learner exit')
        elif id == worker_1_id:
            print('worker_1 exit')
        else:
            print('worker exit!')
        sys.exit()
