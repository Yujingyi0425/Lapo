"""
Based on https://github.com/sfujim/BCQ
离线强化学习的经验回放缓冲区（Replay Buffer）   
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        """
        __init__ 的 Docstring
        经验回放缓冲区初始化    
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param device: PyTorch设备（CPU/GPU）
        :param max_size: 缓冲区最大容量（默认200万条经验）  
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['not_done'] = np.zeros((max_size, 1))

        self.min_r, self.max_r = 0, 0

        self.action_mean = None
        self.action_std = None
        self.state_mean = None
        self.state_std = None

    def add(self, state, action, next_state, reward, done):
        """
        添加经验
        :param self: 说明
        :param state: 当前状态
        :param action:   采取的动作     
        :param next_state: 下一个状态
        :param reward: 奖励
        :param done: 是否结束标志
        """
        self.storage['state'][self.ptr] = state.copy()
        self.storage['action'][self.ptr] = action.copy()
        self.storage['next_state'][self.ptr] = next_state.copy()

        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        sample 的 Docstring
        
        :param self: 说明
        随机采样batch_size条经验
        :param batch_size: 采样批次大小
        :return: 批次经验的状态、动作、下一个状态、奖励和非结束标志
        
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.storage['state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['action'][ind]).to(self.device),
            torch.FloatTensor(self.storage['next_state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['reward'][ind]).to(self.device),
            torch.FloatTensor(self.storage['not_done'][ind]).to(self.device),
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def normalize_state(self, state):
        """
        normalize_state 的 Docstring
        使用均值和标准差进行归一化
        :param self: 标准化状态
        :param state: 输入状态
        :return: 标准化后的状态
        """
        return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        """
        unnormalize_state 的 Docstring
        反归一化恢复原始状态
        :param self: 反归一化状态
        :param state: 标准化状态
        :return: 原始状态
        """
        return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean

    def renormalize(self):
        """
        renormalize 的 Docstring
        :param self: 重新计算均值和标准差并归一化存储的状态和动作
        先反归一化所有存储数据到原始范围
        重新计算均值和标准差
        用新参数重新进行归一化
        同时记录奖励的最小/最大值
        """
        self.storage['state'] = self.unnormalize_state(self.storage['state'])
        self.storage['next_state'] = self.unnormalize_state(self.storage['next_state'])
        self.storage['action'] = self.unnormalize_action(self.storage['action'])

        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)        

        self.storage['state'] = self.normalize_state(self.storage['state'])
        self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
        self.storage['action'] = self.normalize_action(self.storage['action'])

        self.min_r = self.storage['reward'].min()
        self.max_r = self.storage['reward'].max()

    def load(self, data):
        """
        load 的 Docstring
        加载D4RL格式的数据
        从字典格式数据加载（D4RL数据集标准格式）
        期望的键：observations, actions, next_observations, rewards, terminals
        逐条添加到缓冲区
        """
        assert('next_observations' in data.keys())

        for i in range(data['observations'].shape[0]):
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     data['rewards'][i], data['terminals'][i])
                     
        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)

        self.storage['state'] = self.normalize_state(self.storage['state'])
        self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
        self.storage['action'] = self.normalize_action(self.storage['action'])