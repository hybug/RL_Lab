'''
Author: hanyu
Date: 2022-07-19 16:53:16
LastEditTime: 2022-08-04 20:35:41
LastEditors: hanyu
Description: ppo policy
FilePath: /RL_Lab/alogrithm/ppo/ppo_policy.py
'''
from typing import List
import numpy as np
import tensorflow as tf
from configs.config_base import Params
from models.categorical_model import CategoricalModel
from policy.policy_base import PolicyBase
from policy.policy_utils import explained_variance
from policy.sample_batch import SampleBatch
from postprocess.postprocess import Postprocessing


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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                                  epsilon=1e-5)

        self.num_envs = num_envs
        self.nbatch = self.num_envs * self.params.trainer.steps_per_epoch
        self.nbatch_train = self.nbatch // self.params.trainer.num_mini_batches
        assert self.nbatch % self.params.trainer.num_mini_batches == 0

    def update(self, rollouts: SampleBatch) -> dict:
        """Update Policy Network and Value Network

        Args:
            rollouts (dict): obs, act, adv, rets, logp_t

        Returns:
            dict: loss_p, loss_entropy, approx_ent, kl, loss_v, loss_total
        """
        indexs = np.arange(self.nbatch)

        for i in range(self.train_iter):

            losses = self._inner_update_loop(rollouts, indexs)

        ev = explained_variance(rollouts[SampleBatch.VF_PREDS], rollouts[Postprocessing.VALUE_TARGETS])
        losses["explained_variance"] = ev
        return losses

    def _inner_update_loop(self, rollouts: SampleBatch,
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
            # slices = indexs[start:end]
            losses = self._train_one_step(rollouts.slice(start, end))
            means.append([
                losses['policy_loss'], losses['value_loss'],
                losses['entropy_loss'], losses['approx_ent'],
                losses['approx_kl'], losses["clipfrac"]
            ])
        means = np.asarray(means)
        means = np.mean(means, axis=0)
        # ev = explained_variance(prev_fn_out, rets)

        return {
            "policy_loss": means[0],
            "value_loss": means[1],
            "entropy_loss": means[2],
            "approx_ent": means[3],
            "approx_kl": means[4],
            "clipfrac": means[5]
            # "explained_variance": ev
        }

    def _train_one_step(self, mini_rollouts: SampleBatch):
        with tf.GradientTape() as tape:
            _losses = self._loss(mini_rollouts)

        trainable_variables = self.model.trainable_variables()
        # trainable_variables = self.model.base_model.trainable_variables

        # Get gradients
        grads = tape.gradient(_losses['total_loss'], trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grads)

        self.optimizer.apply_gradients(list(zip(grads, trainable_variables)))
        return _losses

    def _loss(self, mini_rollouts: SampleBatch):
        # Depending on the policy(categorical or gaussian)
        # output from the network are logits or mu
        logits, values = self.model({"obs": mini_rollouts[SampleBatch.OBS]})

        logp = self.model.logp(logits, mini_rollouts[SampleBatch.ACTIONS])
        ratio = tf.exp(logp - mini_rollouts[SampleBatch.ACTION_LOGP])
        action_kl = self.model.kl(mini_rollouts[SampleBatch.LOGITS], logits)
        kl_loss = tf.reduce_mean(action_kl)

        pg_loss1 = -mini_rollouts[Postprocessing.ADVANTAGES] * ratio
        pg_loss2 = -mini_rollouts[Postprocessing.ADVANTAGES] * tf.clip_by_value(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # For maxmizing via backprop losses must have negative sign
        policy_loss = tf.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

        entropy = self.model.entropy(logits)
        entropy_loss = tf.reduce_mean(entropy)

        approx_kl = .5 * tf.reduce_mean(tf.square(mini_rollouts[SampleBatch.ACTION_LOGP] - logp))
        approx_ent = tf.reduce_mean(-logp)
        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_ratio),
                    tf.float32))

        # value_loss = tf.reduce_mean(tf.square(rets - values))
        vpred = values
        vpred_clipped = mini_rollouts[SampleBatch.VF_PREDS] + tf.clip_by_value(
            vpred - mini_rollouts[SampleBatch.VF_PREDS], -self.clip_ratio, self.clip_ratio)
        vf_loss1 = tf.math.square(vpred - mini_rollouts[Postprocessing.VALUE_TARGETS])
        vf_loss2 = tf.math.square(vpred_clipped - mini_rollouts[Postprocessing.VALUE_TARGETS])
        value_loss = tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))

        total_loss = policy_loss + self.kl_coef * kl_loss + value_loss * self.v_coef - entropy_loss * self.ent_coef
        # total_loss = policy_loss + value_loss * self.v_coef - entropy_loss * self.ent_coef

        return {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "approx_kl": approx_kl,
            "approx_ent": approx_ent,
            "clipfrac": clipfrac
        }
