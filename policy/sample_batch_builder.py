'''
Author: hanyu
Date: 2022-08-04 11:49:35
LastEditTime: 2022-08-04 14:27:41
LastEditors: hanyu
Description: sample batch builder
FilePath: /RL_Lab/policy/sample_batch_builder.py
'''

import collections
import numpy as np
from typing import Any, Dict, List

from policy.sample_batch import SampleBatch


def to_float_array(v: List[Any]) -> np.ndarray:
    arr = np.array(v)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


class SampleBatchBuilder:
    """Util to build a SampleBatch incrementally. Refer to RLlib implement
    """
    def __init__(self) -> None:
        self.buffers: Dict[str, List] = collections.defaultdict(list)
        self.count = 0

    def add_values(self, **values: Any) -> None:
        """Add the given dictionary (row) of values to this batch.
        """
        for k, v in values.items():
            self.buffers[k].append(v)
        self.count += 1

    def add_batch(self, batch: SampleBatch) -> None:
        """Add the given batch of values to this batch.

        Args:
            batch (SampleBatch): another sample_batch
        """
        for k, column in batch.items():
            self.buffers[k].extend(column)
        self.count += batch.count

    def build_and_reset(self) -> SampleBatch:
        """Returns a sample batch including all previously added values.
        Call after you add all the sample data

        Returns:
            SampleBatch: sample batch
        """
        batch = SampleBatch(
            {k: to_float_array(v)
             for k, v in self.buffers.items()})
        self.buffers.clear()
        self.count = 0
        return batch
