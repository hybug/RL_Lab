'''
Author: hanyu
Date: 2022-07-19 17:45:25
LastEditTime: 2022-08-03 20:15:54
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
from preprocess.feature_encoder.fe_gfootball import FeatureEncoderGFootball
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
            self.params.trainer.nn_architecure)(params=self.params.policy)

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
        if "football" in self.params.env.env_name:
            self.fe = FeatureEncoderGFootball()
        else:
            self.fe = None

        # Initialize rollout worker
        self.rollout_worker = RolloutWorker(
            params=self.params,
            batched_env=self.env,
            model=self.model,
            # feture_encoder=self.fe,
            steps_per_epoch=self.params.trainer.steps_per_epoch,
            gamma=self.params.trainer.gamma,
            lam=self.params.trainer.lam)

        # Initialize the PPO policy
        self.ppo_policy = PPOPolicy(model=self.model,
                                    params=self.params,
                                    num_envs=self.params.env.num_envs)

    def train(self):
        # Starting training
        for epoch in range(self.params.trainer.epochs):
            # Rollout one epoch and get rollout data
            rollouts, episode_infos = self.rollout_worker.rollout()

            # Update the ppo policy
            losses = self.ppo_policy.update(rollouts)

            # Logging training information
            self.logger.store(name="Loss Policy", value=losses["policy_loss"])
            self.logger.store(name="Loss Value", value=losses["value_loss"])
            self.logger.store(name="Loss Entropy", value=losses["entropy_loss"])
            self.logger.store(name="Approx KL", value=losses["approx_kl"])
            self.logger.store(name="Approx Entropy",
                              value=losses["approx_ent"])
            self.logger.store(name="Clip Frac", value=losses["clipfrac"])
            self.logger.store(name="Explained Variance", value=losses["explained_variance"])
            # Logging episode information
            for episode_rew, episode_len in zip(
                    episode_infos["episode_rewards"],
                    episode_infos["episode_lengths"]):
                self.logger.store(name="Episode Reward", value=episode_rew)
                self.logger.store(name="Episode Length", value=episode_len)

            # Save model frequency
            if (epoch + 1) % self.params.trainer.save_freq == 0:
                self.model.save(BASEDIR + f"/saved_models/{self.experiment_name}/checkpoint_{epoch}")

            self.logger.log_metrics(epoch)

        # Close env
        self.env.close()
