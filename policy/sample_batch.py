'''
Author: hanyu
Date: 2022-08-04 11:02:18
LastEditTime: 2022-08-04 11:25:44
LastEditors: hanyu
Description: sample batch
FilePath: /RL_Lab/policy/sample_batch.py
'''

import numpy as np


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
