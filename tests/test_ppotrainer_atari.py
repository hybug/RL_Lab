'''
Author: hanyu
Date: 2022-08-19 15:20:54
LastEditTime: 2022-08-26 15:43:23
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/tests/test_ppotrainer_atari.py
'''

from alogrithm.ppo.ppo_trainer import PPOTrainer
from get_logger import BASEDIR
import numpy as np
import random


def set_global_seeds(i):

    rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def main():
    set_global_seeds(0)
    ppo_trainer = PPOTrainer(
        params_config_path="/config/test_atari/test_BeamRider.yaml",
        working_dir=BASEDIR,
        experiment_name="test_BeamRider_test")
    ppo_trainer.train()
    print()


if __name__ == "__main__":
    main()
