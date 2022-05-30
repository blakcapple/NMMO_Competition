from copy import deepcopy
import time
from abc import ABC, abstractmethod
import pyarrow as pa
import torch
import torch.nn as nn
import zmq
import numpy as np 
import torch 

class Worker(ABC):
    """Abstract class for ApeX distrbuted workers """
    def __init__(
        self, worker_id: int, port_cfg: dict
    ):
        self.worker_id = worker_id
        # unpack communication configs
        self.sub_port = port_cfg["pubsub_port"] # sub params from learner
        self.push_port = port_cfg["pullpush_port"] # push data to learner
        if self.worker_id == 1:
            self.pair_port = port_cfg['pair_port'] # send evaluate data to learner 
        if self.worker_id != 1:
            self.sub_port_2 = port_cfg['pubsub_port2'] # get new info from learner
        # initialize zmq sockets
        print(f"[Worker {self.worker_id}]: initializing sockets..")
        self.initialize_sockets()

        self.learning_stage = 0

    @abstractmethod
    def collect_data(self):
        """Run environment and collect data until stopping criterion satisfied"""
        pass

    @abstractmethod
    def test_run(self):
        """Specifically for the performance-testing worker"""
        pass

    @abstractmethod
    def run(self):
        """main runing loop"""
        pass

    @abstractmethod    
    def synchronize(self, new_params: list):
        """Synchronize worker brain with parameter server"""
        pass 

    @abstractmethod
    def send_replay_data(self, replay_data):
        """
        send collected data to central buffer
        """
    
    @abstractmethod
    def receive_new_params(self):
        """
        receive new params from learner
        """

    def initialize_sockets(self):
        # for receiving params from learner
        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB) # 订阅模式
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅模式订阅的话题
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.sub_port}")

        # for sending replay data to buffer
        time.sleep(1)
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.push_port}")

        if self.worker_id == 1:
            context = zmq.Context()
            self.pair_socket = context.socket(zmq.PAIR)
            self.pair_socket.connect(f"tcp://127.0.0.1:{self.pair_port}")

        if self.worker_id != 1:
            self.sub_socket_2 = context.socket(zmq.SUB) # 订阅模式
            self.sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "stage") # 订阅模式订阅的话题
            # self.sub_socket_2.setsockopt(zmq.CONFLATE, 1)
            self.sub_socket_2.connect(f"tcp://127.0.0.1:{self.sub_port_2}")


class RLWorker(Worker):

    def __init__(
        self, worker_id: int, port_cfg: dict
        ):
        super().__init__(worker_id, port_cfg)

        self.policy = None 

    def synchronize(self, new_params: list):

        new_actor_params, new_critic_params = new_params
        for actor_param, new_actor_params in zip(self.policy.actor.parameters(), new_actor_params):
            new_actor_params = torch.FloatTensor(new_actor_params.copy()).to(self.device)
            actor_param.data.copy_(new_actor_params)
            
        for critic_param, new_critic_params in zip(self.policy.critic.parameters(), new_critic_params):
            new_critic_params = torch.FloatTensor(new_critic_params.copy()).to(self.device)
            critic_param.data.copy_(new_critic_params)

    def send_replay_data(self, replay_data):
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def receive_new_params(self, wait=True):
        new_params_id = False
        try:
            if wait:
                new_params_id = self.sub_socket.recv()
            else:
                new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_id:
            new_params = pa.deserialize(new_params_id)
            self.synchronize(new_params)
            return True

    def receive_new_learning_stage(self):

        new_stage_id = False 
        try:
            new_stage_id = self.sub_socket_2.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if new_stage_id:
            topic, messagedata = new_stage_id.split()
            self.learning_stage = int(messagedata)

    def send_evaluate_data(self, evaluate_data):
        evaluate_data_id = pa.serialize(evaluate_data).to_buffer()
        self.pair_socket.send(evaluate_data_id)



