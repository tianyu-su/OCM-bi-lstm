# -*- coding: utf-8 -*-
# Date: 2021/04/04 11:32

"""

"""
__author__ = 'tianyu'

import numpy as np

# a = [np.array([[1, 2, 3]]),
#      np.array([[4, 5, 6]])]
# print("列表a如下：")
# print(a)
# c = np.stack(a, axis=0)

import json
# cc=json.load(open("/media/tianyu/Data/workspaces/2017MM_BiLSTM/data/benchmark_dataset/label/nondisjoint/fill_in_blank_test.json"))
cc=json.load(open("/media/tianyu/Data/workspaces/2017MM_BiLSTM/data/benchmark_dataset/label/disjoint/fill_in_blank_test.json"))