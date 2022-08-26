'''
Author: hanyu
Date: 2022-07-19 16:53:16
LastEditTime: 2022-08-26 11:44:43
LastEditors: hanyu
Description: ppo policy
FilePath: /RL_Lab/alogrithm/ppo/ppo_policy.py
'''
import numpy as np
import tensorflow as tf
from configs.config_base import Params
from get_logger import TFLogger
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
                 logger: TFLogger,
                 num_envs: int = 1) -> None:
        super().__init__(params)

        self.model = model
        self.logger = logger
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                                  epsilon=1e-7)

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

        self.update_kl(losses["mean_kl"])

        return losses

    def update_kl(self, sampled_kl):
        """Update the current kl coeff baed one the recently measured value

        Args:
            sampled_kl (_type_): meaured value
        """
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coef *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coef *= 0.5
        else:
            return self.kl_coef

        return self.kl_coef

    def _inner_update_loop(self, rollouts: SampleBatch,
                           indexs: np.array) -> dict:
        """Make update with random sampled minibatches and return mean kl-divvergence for early breaking

        Args:
            rollouts (SampleBatch): rollout traject samplebatches
            indexs (np.array): indes

        Returns:
            dict: losses
        """
        np.random.shuffle(indexs)
        # means = list()

        for start in range(0, self.nbatch, self.nbatch_train):
            end = start + self.nbatch_train
            # slices = indexs[start:end]
            losses = self._train_one_step(rollouts.slice(start, end))

            # Store the stats logging info
            for key, value in losses.items():
                self.logger.store(name=key, value=value)

            ev = explained_variance(
                rollouts.slice(start, end)[Postprocessing.VALUE_TARGETS],
                losses["value_fn_out"])
            self.logger.store("explained_variance", ev)

        return losses

    def _train_one_step(self, mini_rollouts: SampleBatch):
        with tf.GradientTape() as tape:
            _losses = self._loss(mini_rollouts)

        trainable_variables = self.model.trainable_variables()
        # trainable_variables = self.model.base_model.trainable_variables

        # Get gradients
        grads = tape.gradient(_losses['total_loss'], trainable_variables)
        # grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grads)

        self.optimizer.apply_gradients(list(zip(grads, trainable_variables)))
        return _losses

    def _loss(self, mini_rollouts: SampleBatch):
        # Depending on the policy(categorical or gaussian)
        # output from the network are logits or mu
        logits, values = self.model.forward(
            {"obs": mini_rollouts[SampleBatch.OBS]})

        logp = self.model.logp(logits, mini_rollouts[SampleBatch.ACTIONS])
        ratio = tf.exp(logp - mini_rollouts[SampleBatch.ACTION_LOGP])
        action_kl = self.model.kl(mini_rollouts[SampleBatch.LOGITS], logits)
        kl_loss = tf.reduce_mean(action_kl)

        pg_loss1 = -mini_rollouts[Postprocessing.ADVANTAGES] * ratio
        pg_loss2 = -mini_rollouts[Postprocessing.ADVANTAGES] * tf.clip_by_value(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        # For maxmizing via backprop losses must have negative sign
        policy_loss = tf.reduce_mean(tf.math.maximum(pg_loss1, pg_loss2))

        entropy = self.model.entropy(logits)
        entropy_loss = tf.reduce_mean(entropy)

        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_param),
                    tf.float32))

        vpred = values
        vf_loss = tf.math.square(vpred -
                                 mini_rollouts[Postprocessing.VALUE_TARGETS])
        vf_loss_clipped = tf.clip_by_value(vf_loss, 0, self.vf_clip_param)
        value_loss = tf.reduce_mean(vf_loss_clipped)

        total_loss = policy_loss + self.kl_coef * kl_loss + value_loss * self.vf_loss_coef - entropy_loss * self.ent_coef

        return {
            "total_loss": total_loss,
            "mean_policy_loss": policy_loss,
            "mean_vf_loss": value_loss,
            "mean_entropy": entropy_loss,
            "mean_kl": kl_loss,
            "value_fn_out": values,
            "clip_frac": clipfrac,
            "kl_coeff": self.kl_coef
        }
