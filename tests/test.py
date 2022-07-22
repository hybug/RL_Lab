'''
Author: hanyu
Date: 2022-07-20 16:40:55
LastEditTime: 2022-07-20 16:46:26
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/tests/test.py
'''
import numpy as np

r = None
a = np.array([[1, 1, 1]])
b = np.array([[2, 2, 2]])
c = np.array([[3, 3, 3]])
for t in [a, b, c]:
    if r is not None:
        r = np.concatenate((r, t), axis=0)
    else:
        r = t
 
print()