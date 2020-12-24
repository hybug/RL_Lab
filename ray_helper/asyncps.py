# coding: utf-8
import time
import ray


@ray.remote(num_cpus=1)
class AsyncPS:
    def __init__(self):
        self._params = None
        self.hashing = int(time.time())

    def push(self, params):
        self._params = params
        self.hashing = int(time.time())

    def pull(self):
        return self._params

    def get_hashing(self):
        return self.hashing
