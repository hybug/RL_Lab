'''
Author: hanyu
Date: 2024-05-15 20:41:29
LastEditTime: 2024-05-15 21:12:13
LastEditors: hanyu
Description: trainer
FilePath: /RL_Lab/alogrithm/alphazero/alphazero_trainer.py
'''

import torch

from networks.torch_model import Resnet


class Trainer:
    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        self.loss = None
        self.device = None

        self.init_model()

    def init_model(self):
        self.model = Resnet()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss = torch.nn.CrossEntropyLoss()

    