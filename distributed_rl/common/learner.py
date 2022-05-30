from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import pyarrow as pa
import torch.nn as nn
import zmq

class Learner(ABC):
    def __init__(
        self, port_cfg: dict,
    ):
    
        # unpack communication configs
        self.pull_port = port_cfg["pullpush_port"] # pull data from worker
        self.pub_port = port_cfg["pubsub_port"] # pub params to worker
        self.pair_port = port_cfg['pair_port'] # recieve data from worker 
        self.pub_port_2 = port_cfg['pubsub_port2'] # send extra info to worker

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        self.initialize_sockets()

    @abstractmethod
    def get_params(self):
        """Return model params for synchronization"""
        pass

    # @abstractmethod
    # def load_weights(self, load_dict:dict):
    #     """
    #     载入预训练的模型参数
    #     """
    #     pass
    
    # @abstractmethod
    # def save_weights(self, save_dict:dict):
    #     """
    #     保存训练中的模型参数
    #     """
    #     pass

    @abstractmethod
    def publish_params(self, new_params: np.ndarray):
        """
        发布模型参数
        """

    @abstractmethod
    def recv_replay_data_(self):
        """
        接收replay data到缓存队列
        """

    @abstractmethod
    def run(self):
        """
        主运行函数
        """

    def initialize_sockets(self):
        """
        初始话通信接口
        """
        # For sending new params to workers
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pub_port}")

        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.pull_port}")

        context = zmq.Context()
        self.pair_socket = context.socket(zmq.PAIR)
        self.pair_socket.bind(f'tcp://127.0.0.1:{self.pair_port}')

        context = zmq.Context()
        self.pub_socket_2 = context.socket(zmq.PUB)
        self.pub_socket_2.bind(f"tcp://127.0.0.1:{self.pub_port_2}")






    


