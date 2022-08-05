'''
Author: hanyu
Date: 2022-07-19 16:14:35
LastEditTime: 2022-08-05 10:39:04
LastEditors: hanyu
Description: rollout worker
FilePath: /RL_Lab/workers/rollout_worker.py
'''
from typing import Callable, Tuple

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

        _sample_batch_builders = [
            SampleBatchBuilder() for _ in range(self.params.env.num_envs)
        ]
        # ep_rews, ep_lens = list(), list()

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

            # obses.append(self.fe.transform(self._obs))
            # dones.append(self._dones)
            # actions.append(actions_t)
            # values.append(values_t)
            # logits.append(logits_t)
            # logp.append(logp_t)

            # Envs stepping actions
            self._obs_dict, self._rews_dict, self._dones_dict, infos_dict = self.batched_env.step(
                actions_t)

            # for info in infos:
            #     if info and "score_reward" not in info.keys():
            #         ep_rews.append(info['episode_reward'])
            #         ep_lens.append(info['episode_length'])

            # Save rews(t+1) into sample_batch
            for w_i in range(self.params.env.num_envs):
                _sample_batch_builders[w_i].add_values(
                    rewards=self._rews_dict[w_i], infos=infos_dict[w_i])
            # rews.append(self._rews)

        # In end for steps_per_epoch
        # Get the last Values for BOOTSTRAPING
        # Or complete the leftover steps& cutoff the trajectory outside step_per_epoch TODO

        # Model infer the last actions TODO
        _, _, last_values, _ = self.model.get_action_logp_value(
            {"obs": self.fe.transform(self._obs_dict)})
        # last_values = values_t

        # obses = np.array(obses, dtype=np.float32)
        # rews = np.array(rews, dtype=np.float32)
        # dones = np.array(dones, dtype=np.bool)
        # actions = np.array(actions, dtype=np.float32)
        # values = np.array(values, dtype=np.float32)
        # logits = np.array(logits, dtype=np.float32)
        # logp = np.array(logp, dtype=np.float32)
        batches = [s_b_b.build_and_reset() for s_b_b in _sample_batch_builders]

        for i, rollout_batch in enumerate(batches):
            assert rollout_batch.count == self.steps_per_epoch
            rollout_batch = compute_gae_from_sample_batch(
                rollout=rollout_batch,
                last_value=last_values[i],
                last_done=self._dones_dict[i],
                gamma=self.gamma,
                lam=self.lam)

        batches = SampleBatch.concat_samples(batches)

        # # Discount / Bootstraping Values
        # # And calculate Advantages
        # rets = np.zeros_like(rews)
        # advs = np.zeros_like(rews)
        # last_gae_lam = 0

        # for t in reversed(range(self.steps_per_epoch)):
        #     if t == self.steps_per_epoch - 1:
        #         # The last sampling step
        #         next_non_terminal = np.array(
        #             [1
        #              for _ in range(self.batched_env.num_envs)]) - self._dones
        #         next_values = last_values
        #     else:
        #         next_non_terminal = np.array(
        #             [1
        #              for _ in range(self.batched_env.num_envs)]) - dones[t + 1]
        #         next_values = values[t + 1]

        #     delta = rews[
        #         t] + self.gamma * next_values * next_non_terminal - values[t]
        #     advs[
        #         t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        # rets = advs + values
        # # Normalize advantages
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return batches

    def _flatten_rollout(self, obses: np.array, rews: np.array,
                         dones: np.array, actions: np.array, logp: np.array,
                         values: np.array, advs: np.array, rets: np.array,
                         logits: np.array) -> dict:
        obses = obses.reshape(
            (-1, ) + obses.shape[-len(self.params.policy.input_shape):])
        logits = logits.reshape((-1, ) + logits.shape[-1:])
        return {
            "obses": obses,
            "rews": rews.flatten(),
            "dones": dones.flatten(),
            "actions": actions.flatten(),
            "logp": logp.flatten(),
            "values": values.flatten(),
            "advs": advs.flatten(),
            "returns": rets.flatten(),
            "logits": logits
        }
