'''
Author: hanyu
Date: 2022-07-19 16:57:15
LastEditTime: 2022-07-19 17:46:05
LastEditors: hanyu
Description: 
FilePath: /RL_Lab/trainers/trainer_base.py
'''


class TrainerBase():
    def __init__(
            self,
            params_config_path: str(),
    ) -> None:
        self.params_config_path = params_config_path

    def train(self):
        raise NotImplementedError
