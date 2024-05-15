'''
Author: hanyu
Date: 2022-07-19 16:21:01
LastEditTime: 2022-08-03 15:20:35
LastEditors: hanyu
Description: batched env
FilePath: /RL_Lab/envs/batched_env.py
'''

from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any

import cloudpickle
from loguru import logger


class BatchedEnvBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractproperty
    def num_envs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        pass


class BatchedEnv(BatchedEnvBase):
    def __init__(self, env_fns: list) -> None:
        super().__init__()

        self._process = list()
        self._command_queues = list()
        self._result_queues = list()
        self._env_fns = env_fns

        # Initialize and start worker processes for each env
        for idx, env_fn in enumerate(self._env_fns):
            cmd_queue = Queue()
            res_queue = Queue()
            proc = Process(target=self._worker_process,
                           args=(cmd_queue, res_queue,
                                 cloudpickle.dumps(env_fn), idx))
            proc.start()
            self._process.append(proc)
            self._command_queues.append(cmd_queue)
            self._result_queues.append(res_queue)

    # TODO to be implemented with iter like rllib.rolloutworker
    @staticmethod
    def _worker_process(cmd_queue: Queue, res_queue: Queue,
                        env_pickled_str: str, worker_idx: int):
        """Worker for environment, Implements a loop for waiting and executing commands from the cmd-queue

        Args:
            cmd_queue (Queue): commands queue to be executed
            res_queue (Queue): result queue
            env_pickled_str (str): env_fn
        """
        try:
            # Load lambda create_single_env function and execute it
            env = cloudpickle.loads(env_pickled_str)()
            # res_queue.put((env.env_info, None))

            # Initialize the episode reward and length
            episode_reward = 0
            episode_length = 0

            try:
                while True:

                    cmd, arg = cmd_queue.get()
                    if cmd == 'reset':
                        # Initialize the episode reward and length after reset
                        episode_reward = 0
                        episode_length = 0
                        obs = env.reset()
                        # for _ in range(2):
                        #     obs, _, _, _ = env.step(5)
                        res_queue.put(((obs, 0, False, None), None))

                    elif cmd == 'step':
                        obs, rew, done, info = env.step(arg)

                        # Update the episode reward and length
                        episode_reward += rew
                        episode_length += 1

                        if done:
                            info = {
                                "episode_reward": episode_reward,
                                "episode_length": episode_length
                            }
                            # Reinitialize the episode reward and length after done
                            episode_reward = 0
                            episode_length = 0

                            # env_core = env._env.env.env.env.env.env.env._env
                            # logger.info(
                            #     f"Episode reward: {env_core._cumulative_reward}, score: {env_core._observation['score']}, steps: {env_core._step_count}"
                            # )

                            obs = env.reset()
                            # for _ in range(2):
                            #     obs, _, _, _ = env.step(5)

                        res_queue.put(((obs, rew, done, info), None))

                    elif cmd == 'close':
                        return
            finally:
                env.close()
        except Exception:
            import traceback
            traceback.print_exc()
            # Put exception into the result queue
            res_queue.put(('exception', traceback.format_exc()), None)

    @staticmethod
    def _queue_get(queue: Queue) -> Any:
        value, exception = queue.get(timeout=20)
        if exception is not None:
            raise exception
        return value

    @property
    def num_envs(self):
        return len(self._process)

    def reset(self):
        obses = list()
        rewards = list()
        dones = list()

        for q in self._command_queues:
            q.put(('reset', None))

        for q in self._result_queues:
            obs, reward, done, info = self._queue_get(q)

            obses.append(obs)
            rewards.append(reward)
            dones.append(done)

        return obses, rewards, dones, {}

    def step(self, actions: list):
        obses = list()
        rewards = list()
        dones = list()
        infos = list()

        for q, action in zip(self._command_queues, actions):
            q.put(('step', action))

        for q in self._result_queues.copy():
            try:
                obs, reward, done, info = self._queue_get(q)
            except Empty:
                pass
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return obses, rewards, dones, infos

    def close(self):
        for q in self._command_queues:
            q.put(('close', None))
        for proc in self._process:
            proc.join()
