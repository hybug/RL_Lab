'''
Author: hanyu
Date: 2022-08-04 15:36:13
LastEditTime: 2022-08-04 16:28:08
LastEditors: hanyu
Description: no feature encoder
FilePath: /RL_Lab/preprocess/feature_encoder/fe_no.py
'''
from typing import List
import numpy as np


class FeatureEncoder:
    def __init__(self) -> None:
        pass

    def transform(self):
        raise NotImplementedError


class NoFeatureEncoder:
    def __init__(self) -> None:
        self.name = "NofeatureEncoder"

    def transform(self, observation: List[np.ndarray]) -> np.ndarray:
        if isinstance(observation, list):
            return np.array(observation)
        elif isinstance(observation, dict):
            return np.array(list(observation.values()))
        return observation
