# -*- coding: utf-8 -*-
# Date: 2021/04/03 11:05

"""

"""
__author__ = 'tianyu'

import json
from collections import Counter

splits = ["disjoint", "nondisjoint"]
file_types = ['train', 'valid', 'test']
file_path = "/media/tianyu/Data/workspaces/2017MM_BiLSTM/data/benchmark_dataset/label/{which_split}/{ty_file}_no_dup.json"

words = list()
for sp in splits:
    for fp in file_types:
        with open(file_path.format(which_split=sp, ty_file=fp)) as tt:
            tmp = json.load(tt)
            for one in tmp:
                for it in one["items"]:
                    words.extend(it["name"].split())


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


conter = Counter(words)
sorted_cnt = sorted(conter.items(), key=lambda x: -x[-1])
# cc = filter(lambda x: not is_number(x[0]) and len(x[0]) > 3 and x[-1] > 30, Counter(words).items())
# cc = filter(lambda x:x[-1] > 30, )
with open("../benchmark_final_word_dict.txt", 'w') as fp:
    for dt in sorted_cnt[:2750]:
        fp.write("{}    {}\n".format(*dt))
