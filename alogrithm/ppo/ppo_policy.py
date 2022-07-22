'''
Author: hanyu
Date: 2022-07-19 16:53:16
LastEditTime: 2022-07-20 18:04:01
LastEditors: hanyu
Description: ppo policy
FilePath: /RL_Lab/alogrithm/ppo/ppo_policy.py
'''
import numpy as np
import tensorflow as tf
from loguru import logger
from configs.config_base import Params
from models.categorical_model import CategoricalModel
from policy.policy_base import PolicyBase


class PPOPolicy(PolicyBase):
    """Proximal Policy Optimization

    Args:
        PolicyBase (class): base policy class
    """
    def __init__(self,
                 model: CategoricalModel,
                 params: Params,
                 num_envs: int = 1) -> None:
        super().__init__(params)

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.num_envs = num_envs
        self.nbatch = self.num_envs * self.params.trainer.steps_per_epoch
        self.nbatch_train = self.nbatch // self.params.trainer.num_mini_batches
        assert self.nbatch % self.params.trainer.num_mini_batches == 0

    def update(self, rollouts: dict) -> dict:
        """Update Policy Network and Value Network

        Args:
            rollouts (dict): obs, act, adv, rets, logp_t

        Returns:
            dict: loss_p, loss_entropy, approx_ent, kl, loss_v, loss_total
        """
        indexs = np.arange(self.nbatch)

        for i in range(self.train_iter):

            losses = self._inner_update_loop(rollouts["obses"],
                                             rollouts["actions"],
                                             rollouts["advs"],
                                             rollouts["returns"],
                                             rollouts["logp"], indexs)
            if losses["approx_kl"] > 1.5 * self.target_kl:
                logger.info(
                    f"Eargly breaking at step{i} due to reaching max kl.")
                break
        return losses

    def _inner_update_loop(self, obses: np.array, actions: np.array,
                           advs: np.array, rets: np.array, logp_t: np.array,
                           indexs: np.array) -> dict:
        """Make update with random sampled minibatches and return mean kl-divvergence for early breaking

        Args:
            obses (np.array): observations
            actions (np.array): actions
            advs (np.array): advantages
            rets (np.array): returns
            logp_t (np.array): logp_t
            indexs (np.array): indexs

        Returns:
            dict: policy_loss, value_loss, entropy_loss, approx_ent, approx_kl
        """
        np.random.shuffle(indexs)
        means = list()

        for start in range(0, self.nbatch, self.nbatch_train):
            end = start + self.nbatch_train
            slices = indexs[start:end]
            losses = self._train_one_step(obses[slices], actions[slices],
                                          advs[slices], logp_t[slices],
                                          rets[slices])
            means.append([
                losses['policy_loss'], losses['value_loss'],
                losses['entropy_loss'], losses['approx_ent'],
                losses['approx_kl']
            ])
        means = np.asarray(means)
        means = np.mean(means, axis=0)

        return {
            "policy_loss": means[0],
            "value_loss": means[1],
            "entropy_loss": means[2],
            "approx_ent": means[3],
            "approx_kl": means[4]
        }

    def _train_one_step(self, obs, act, adv, logp_old, rets):
        with tf.GradientTape() as tape:
            _losses = self._loss(obs, logp_old, act, adv, rets)

        trainable_variables = self.model.trainable_variables

        # Get gradients
        grads = tape.gradient(_losses['total_loss'], trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grads)

        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return _losses

    def _loss(self, obs, logp_old, act, adv, rets):
        # Depending on the policy(categorical or gaussian)
        # output from the network are logits or mu
        logits, values = self.model({"obs": obs})

        logp = self.model.logp(logits, act)
        ratio = tf.exp(logp - logp_old)
        min_adv = tf.where(adv > 0, (1 + self.clip_ratio) * adv,
                           (1 - self.clip_ratio) * adv)

        # Policy Loss = loss_clipped + loss_entropy * entropy_coeff
        # For maxmizing via backprop losses must have negative sign
        clipped_loss = -tf.reduce_mean(tf.math.minimum(ratio * adv, min_adv))

        entropy = self.model.entropy(logits)
        entropy_loss = -tf.reduce_mean(entropy)

        policy_loss = clipped_loss + entropy_loss * self.ent_coef

        approx_kl = tf.reduce_mean(logp_old - logp)
        approx_ent = tf.reduce_mean(-logp)

        value_loss = 0.5 * tf.reduce_mean(tf.square(rets - values))

        total_loss = policy_loss + value_loss * self.v_coef

        return {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "approx_kl": approx_kl,
            "approx_ent": approx_ent
        }
