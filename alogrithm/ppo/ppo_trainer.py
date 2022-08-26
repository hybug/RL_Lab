'''
Author: hanyu
Date: 2022-07-19 17:45:25
LastEditTime: 2022-08-26 11:07:12
LastEditors: hanyu
Description: ppo trainer
FilePath: /RL_Lab/alogrithm/ppo/ppo_trainer.py
'''
import numpy as np
import tensorflow as tf
from alogrithm.ppo.ppo_config import PPOParams
from alogrithm.ppo.ppo_policy import PPOPolicy
from envs.env_utils import create_batched_env
from get_logger import BASEDIR, TFLogger
from models.categorical_model import CategoricalModel
from networks.network_utils import nn_builder
from policy.sample_batch import SampleBatch
from preprocess.feature_encoder.fe_no import NoFeatureEncoder
from trainers.trainer_base import TrainerBase
from workers.rollout_worker import RolloutWorker


class PPOTrainer(TrainerBase):
    def __init__(self,
                 params_config_path: str,
                 working_dir: str = BASEDIR,
                 experiment_name: str = "test") -> None:
        super().__init__(params_config_path)
        self.experiment_name = experiment_name

        # Init trainer params
        self.params = PPOParams(config_file=self.params_config_path)

        # Set random seed for tensorflow and numpy
        tf.random.set_seed(self.params.trainer.seed)
        np.random.seed(self.params.trainer.seed)

        # Create batched environment in multiprocessing mode
        self.env = create_batched_env(self.params.env.num_envs,
                                      self.params.env)

        # Initialize Logger
        self.logger = TFLogger(experiment_name, working_dir)

        # Initialize nerual network
        self.network = nn_builder(
            self.params.policy.nn_architecure)(params=self.params.policy)

        # Initialize model
        if not self.params.env.is_act_continuous:
            model_cls = CategoricalModel
        else:
            raise NotImplementedError("Error >> Model not implemented")
        self.model = model_cls(network=self.network, params=self.params)

        # Restore model
        if self.params.trainer.restore_path:
            raise NotImplementedError("Error >> Restore model not implemented")

        # Initialize feature encoder
        self.fe = NoFeatureEncoder(params=self.params)

        # Initialize rollout worker
        self.rollout_worker = RolloutWorker(
            params=self.params,
            batched_env=self.env,
            model=self.model,
            feature_encoder=self.fe,
            steps_per_epoch=self.params.trainer.steps_per_epoch,
            gamma=self.params.trainer.gamma,
            lam=self.params.trainer.lam)

        # Initialize the PPO policy
        self.ppo_policy = PPOPolicy(model=self.model,
                                    params=self.params,
                                    logger=self.logger,
                                    num_envs=self.params.env.num_envs)

    def train(self):
        # Starting training
        for epoch in range(self.params.trainer.epochs):
            # Rollout one epoch and get rollout data
            rollout_sample_batches = self.rollout_worker.rollout()

            # Update the ppo policy
            _ = self.ppo_policy.update(rollout_sample_batches)

            # Logging episode information
            self.logger.store("ts", (epoch + 1) *
                              self.params.trainer.steps_per_epoch * self.params.env.num_envs)
            for info in rollout_sample_batches[SampleBatch.INFOS]:
                if "episode_reward" in info.keys():
                    self.logger.store(name="Episode Reward",
                                      value=info["episode_reward"])
                if "episode_length" in info.keys():
                    self.logger.store(name="Episode Length",
                                      value=info["episode_length"])

            # Save model frequency
            if (epoch + 1) % self.params.trainer.save_freq == 0:
                self.model.save(
                    BASEDIR +
                    f"/saved_models/{self.experiment_name}/checkpoint_{epoch}")

            self.logger.log_metrics(epoch)

        # Close env
        self.env.close()
