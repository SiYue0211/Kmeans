#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 18:33
# @Author  : dawn
# @File    : test.py

import numpy as np

cluster = np.array([[1, 0, 2, 1, 0],[3, 4, 5, 6, 4]])
cluster = np.transpose(cluster)
print(cluster)
k_index = np.where(cluster[:, 0] == 1)[0]
K_value = cluster[k_index, :]
mean = np.mean(K_value, axis=0)
print(mean)