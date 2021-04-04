# -*- coding: utf-8 -*-
# Date: 2021/04/03 16:58

"""

"""
__author__ = 'tianyu'

import json

file_types = ['train', 'valid', 'test']
# fn_path = "../../label/{}_no_dup.json"
fn_path = "../label/disjoint/{}_no_dup.json"
# fn_path = "../label/nondisjoint/{}_no_dup.json"

all_sentences = set()
for fyp in file_types:
    with open(fn_path.format(fyp)) as fp:
        fn = json.load(fp)
        print(len(fn))
    all_sentences.update([it["name"] for one in fn for it in one["items"]])

all_sentences.remove("")

