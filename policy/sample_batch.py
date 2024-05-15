'''
Author: hanyu
Date: 2022-08-04 11:02:18
LastEditTime: 2022-08-26 11:45:55
LastEditors: hanyu
Description: sample batch
FilePath: /RL_Lab/policy/sample_batch.py
'''

from typing import List
import numpy as np


# https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py
class SampleBatch(dict):
    """Similar to Rllib's SampleBatch class
    """

    # Outputs from interacting with the environment.
    OBS = "obs"
    CUR_OBS = "obs"
    NEXT_OBS = "new_obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    PREV_ACTIONS = "prev_actions"
    PREV_REWARDS = "prev_rewards"
    DONES = "dones"
    INFOS = "infos"

    # Extra action fetches keys.
    LOGITS = "logits"
    ACTION_DIST_INPUTS = "action_dist_inputs"
    ACTION_PROB = "action_prob"
    ACTION_LOGP = "action_logp"

    # Uniquely identifies an episode.
    EPS_ID = "eps_id"

    # Uniquely identifies a sample batch. This is important to distinguish RNN
    # sequences from the same episode when multiple sample batches are
    # concatenated (fusing sequences across batches can be unsafe).
    UNROLL_ID = "unroll_id"

    # Uniquely identifies an agent within an episode.
    AGENT_INDEX = "agent_index"

    # Value function predictions emitted by the behaviour policy.
    VF_PREDS = "vf_preds"

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

        lengths = list()
        copy_ = {k: v
                 for k, v in self.items()
                 if k != "seq_lens"}  # TODO RNN implement
        for k, v in copy_.items():
            assert isinstance(k, str), self
            len_ = len(v) if isinstance(v, (list, np.ndarray)) else None
            lengths.append(len_)
            if isinstance(v, list):
                self[k] = np.array(v)

        self.count = lengths[0] if lengths else 0

    @staticmethod
    def concat_samples(samples: List["SampleBatch"]) -> "SampleBatch":
        """Concatenates SampleBatches

        Args:
            samples (List[&quot;SampleBatch&quot;]): List of SampleBatches

        Returns:
            SampleBatch: SampleBatch
        """
        concat_samples = list()
        for s in samples:
            if s.count > 0:
                concat_samples.append(s)

        concatd_data = dict()
        for k in concat_samples[0].keys():
            concatd_data[k] = np.concatenate([s[k] for s in concat_samples])

        return SampleBatch(concatd_data)

    def slice(self, start: int, end: int) -> "SampleBatch":
        return SampleBatch({k: v[start:end] for k, v in self.items()})
