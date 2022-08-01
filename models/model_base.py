'''
Author: hanyu
Date: 2022-07-26 11:08:21
LastEditTime: 2022-07-29 16:32:34
LastEditors: hanyu
Description: model base
FilePath: /RL_Lab/models/model_base.py
'''

from typing import Any, List


class TFModelBase:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.var_list = list()

    def register_variables(self, variables: List[Any]) -> None:
        self.var_list.extend(variables)

    def variables(self) -> list:
        return list(self.var_list)

    def trainable_variables(self):
        return [v for v in self.variables() if v.trainable]

    def forward(self, inputs_dict: dict):
        raise NotImplementedError

    def __call__(self, inputs_dict: dict):
        outputs = self.forward(inputs_dict=inputs_dict)
        return outputs

    def save(self, save_path: str):
        self.base_model.save(save_path)

    def load(self, load_path: str):
        self.base_model.load(load_path)

        def forward(inputs: dict):
            logits, value = self.base_model(inputs['obs'])
            return logits, value

        self.forward_func = forward
