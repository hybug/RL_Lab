'''
Author: hanyu
Date: 2022-07-19 16:14:35
LastEditTime: 2022-08-26 17:53:32
LastEditors: hanyu
Description: rollout worker
FilePath: /RL_Lab/workers/rollout_worker.py
'''
from typing import Tuple

import numpy as np
from configs.config_base import Params
from envs.batched_env import BatchedEnv
from models.model_base import TFModelBase
from policy.sample_batch import SampleBatch
from policy.sample_batch_builder import SampleBatchBuilder
from postprocess.postprocess import compute_gae_from_sample_batch
from preprocess.feature_encoder.fe_no import FeatureEncoder


class RolloutWorker:
    def __init__(self,
                 params: Params,
                 batched_env: BatchedEnv,
                 model: TFModelBase,
                 steps_per_epoch: int,
                 feature_encoder: FeatureEncoder,
                 gamma: float = 0.99,
                 lam: float = 0.97) -> None:
        self.batched_env = batched_env
        self.params = params
        self.model = model
        self.fe = feature_encoder
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.lam = lam

        self._obs_dict = None
        self._rews_dict = None
        self._dones_dict = None
        self._first_reset = False

    def rollout(self) -> Tuple[dict, dict]:
        if not self._first_reset:
            self._obs_dict, self._rews_dict, self._dones_dict, _ = self.batched_env.reset(
            )
            self._first_reset = True

        # Init sample_batch_builder for every env
        _sample_batch_builders = [
            SampleBatchBuilder() for _ in range(self.params.env.num_envs)
        ]

        for step in range(self.steps_per_epoch):

            # Model infer actions
            actions_t, logp_t, values_t, logits_t = self.model.get_action_logp_value(
                {"obs": self.fe.transform(self._obs_dict)})

            assert len(actions_t) == len(logp_t) == len(values_t) == len(
                logits_t) == self.params.env.num_envs

            # Save obs(t), dones(t), actions(t), logp(t), values(t) into sample_batch
            for w_i in range(self.params.env.num_envs):
                _sample_batch_builders[w_i].add_values(
                    agent_index=w_i,
                    obs=self.fe.transform(self._obs_dict[w_i]),
                    actions=actions_t[w_i],
                    dones=self._dones_dict[w_i],
                    vf_preds=values_t[w_i],
                    logits=logits_t[w_i],
                    action_logp=logp_t[w_i])

            # Envs stepping actions
            self._obs_dict, self._rews_dict, self._dones_dict, infos_dict = self.batched_env.step(
                actions_t)

            # Save rews(t+1) into sample_batch
            for w_i in range(self.params.env.num_envs):
                _sample_batch_builders[w_i].add_values(
                    rewards=self._rews_dict[w_i], infos=infos_dict[w_i])
            # rews.append(self._rews)

        # In end for steps_per_epoch
        # Get the last Values for BOOTSTRAPING
        # Or complete the leftover steps& cutoff the trajectory outside step_per_epoch TODO

        # Model infer the last actions
        _, _, last_values, _ = self.model.get_action_logp_value(
            {"obs": self.fe.transform(self._obs_dict)})

        batches = [s_b_b.build_and_reset() for s_b_b in _sample_batch_builders]

        for i, rollout_batch in enumerate(batches):
            rollout_batch[SampleBatch.REWARDS] = np.sign(
                rollout_batch[SampleBatch.REWARDS])
            assert rollout_batch.count == self.steps_per_epoch
            rollout_batch = compute_gae_from_sample_batch(
                rollout=rollout_batch,
                last_value=last_values[i],
                last_done=self._dones_dict[i],
                gamma=self.gamma,
                lam=self.lam)

        batches = SampleBatch.concat_samples(batches)

        return batches
