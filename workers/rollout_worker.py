'''
Author: hanyu
Date: 2022-07-19 16:14:35
LastEditTime: 2022-08-03 19:37:22
LastEditors: hanyu
Description: rollout worker
FilePath: /RL_Lab/workers/rollout_worker.py
'''
from typing import Tuple

import numpy as np
import tensorflow as tf
from envs.batched_env import BatchedEnv


class RolloutWorker:
    def __init__(self,
                 batched_env: BatchedEnv,
                 model: tf.keras.Model,
                 steps_per_epoch: int,
                 feture_encoder=None,
                 gamma: float = 0.99,
                 lam: float = 0.97) -> None:
        self.batched_env = batched_env
        self.model = model
        self.fe = feture_encoder
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.lam = lam

        self._obs = None
        self._rews = None
        self._dones = None
        self._first_reset = False

    def rollout(self) -> Tuple[dict, dict]:
        if not self._first_reset:
            self._obs, self._rews, self._dones, _ = self.batched_env.reset()
            self._first_reset = True

        obses, rews, dones = list(), list(), list()
        actions, values, logp, logits = list(), list(), list(), list()
        ep_rews, ep_lens = list(), list()

        for step in range(self.steps_per_epoch):

            # Model infer actions TODO
            if self.fe:
                obs = self.fe.encode(self._obs)
                obs = self.fe.concate_observation_from_raw(obs)
            else:
                # obs = np.array(self._obs) / 255
                obs = np.array(self._obs)
            actions_t, logp_t, values_t, logits_t = self.model.get_action_logp_value(
                {"obs": obs})

            obses.append(obs.copy())
            dones.append(self._dones)
            actions.append(actions_t)
            values.append(values_t)
            logits.append(logits_t)
            logp.append(logp_t)

            # Envs stepping actions
            self._obs, self._rews, self._dones, infos = self.batched_env.step(
                actions_t)

            for info in infos:
                if info and "score_reward" not in info.keys():
                    ep_rews.append(info['episode_reward'])
                    ep_lens.append(info['episode_length'])
            # Save obs(t), dones(t), actions(t), logp(t), values(t), rews(t+1) in one stepping
            rews.append(self._rews)

        # In end for steps_per_epoch
        # Get the last Values for BOOTSTRAPING
        # Or complete the leftover steps& cutoff the trajectory outside step_per_epoch TODO

        # Model infer the last actions TODO
        if self.fe:
            obs = self.fe.encode(self._obs)
            obs = self.fe.concate_observation_from_raw(obs)
        else:
            obs = np.array(self._obs)
        #         obs = np.array(self._obs) / 255
        _, _, values_t, _ = self.model.get_action_logp_value(
            {"obs": obs})
        last_values = values_t

        obses = np.array(obses, dtype=np.float32)
        rews = np.array(rews, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        actions = np.array(actions, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        logits = np.array(logits, dtype=np.float32)
        logp = np.array(logp, dtype=np.float32)

        # Discount / Bootstraping Values
        # And calculate Advantages
        # TODO waiting for abstract encapsulation
        rets = np.zeros_like(rews)
        advs = np.zeros_like(rews)
        last_gae_lam = 0

        for t in reversed(range(self.steps_per_epoch)):
            if t == self.steps_per_epoch - 1:
                # The last sampling step
                next_non_terminal = np.array(
                    [1
                     for _ in range(self.batched_env.num_envs)]) - self._dones
                next_values = last_values
            else:
                next_non_terminal = np.array(
                    [1
                     for _ in range(self.batched_env.num_envs)]) - dones[t + 1]
                next_values = values[t + 1]

            delta = rews[
                t] + self.gamma * next_values * next_non_terminal - values[t]
            advs[
                t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam

        rets = advs + values
        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return self._flatten_rollout(obses, rews, dones, actions, logp, values,
                                     advs, rets, logits), {
                                         "episode_rewards": ep_rews,
                                         "episode_lengths": ep_lens
                                     }

    def _flatten_rollout(self, obses: np.array, rews: np.array,
                         dones: np.array, actions: np.array, logp: np.array,
                         values: np.array, advs: np.array,
                         rets: np.array, logits: np.array) -> dict:
        obses = obses.reshape((-1, ) + obses.shape[-3:])
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
