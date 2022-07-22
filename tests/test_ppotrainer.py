'''
Author: hanyu
Date: 2022-07-20 14:58:57
LastEditTime: 2022-07-20 15:47:22
LastEditors: hanyu
Description: test ppo trainer
FilePath: /RL_Lab/tests/test_ppotrainer.py
'''

from alogrithm.ppo.ppo_trainer import PPOTrainer
from get_logger import BASEDIR


def main():
    ppo_trainer = PPOTrainer(params_config_path="/config/test/test.yaml",
                             working_dir=BASEDIR,
                             experiment_name="test_ppotrainer")
    ppo_trainer.train()
    print()


if __name__ == "__main__":
    main()
