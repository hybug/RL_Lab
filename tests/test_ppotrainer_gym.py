'''
Author: hanyu
Date: 2022-07-20 14:58:57
LastEditTime: 2022-08-03 16:20:19
LastEditors: hanyu
Description: test ppo trainer
FilePath: /RL_Lab/tests/test_ppotrainer_gym.py
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
    # set_global_seeds(0)
    ppo_trainer = PPOTrainer(
        params_config_path="/config/test_gym/test_CartPole-v1.yaml",
        working_dir=BASEDIR,
        experiment_name="test_CartPole-v1")
    ppo_trainer.train()
    print()


if __name__ == "__main__":
    main()
