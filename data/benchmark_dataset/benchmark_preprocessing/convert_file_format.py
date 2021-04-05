# -*- coding: utf-8 -*-
# Date: 2021/04/03 11:04

"""

"""
__author__ = 'tianyu'

import itertools
import os.path as osp
import json

import random

# which_split = "nondisjoint"
which_split = "disjoint"

root_dir = "/media/tianyu/Software/workspaces/datasets/Fashion/polyvore/polyvore_outfits"
meta_path = osp.join(root_dir, "polyvore_item_metadata.json")
src_file = osp.join(root_dir, which_split, "{}.json")
compatibility_file = osp.join(root_dir, which_split, "compatibility_test.txt")
FITB_file = osp.join(root_dir, which_split, "fill_in_blank_test.json")
retrieval_file = osp.join(root_dir, "retrival_data", "same_type", "{}.json".format(which_split))
output_file = osp.join("../label", which_split, "{}_no_dup.json")
output_cp = osp.join("../label", which_split, "fashion_compatibility_prediction.txt")
output_fitb = osp.join("../label", which_split, "fill_in_blank_test.json")
output_retreival = osp.join("../label", which_split, "retrieval_test.json")
output_retreival_features = osp.join("../label", which_split, "retrieval_test_features.json")
output_set2id_mapping = osp.join("../label", which_split, "set2id_mapping.json")
file_types = ['train', 'valid', 'test']


def generate_benchmark_setid2imid():
    mapping = {}
    for ty in file_types:
        with open(src_file.format(ty)) as fp:
            cur_file = json.load(fp)
        for one in cur_file:
            set_id = one["set_id"]
            for it in one["items"]:
                mapping["{}_{}".format(set_id, it["index"])] = it["item_id"]
    with open(output_set2id_mapping, 'w') as fp:
        json.dump(mapping, fp)


def squeeze_benchmark_idx():
    old_2_new_setid_mapping = {}
    tot_set_id = {kk: [] for kk in file_types}
    for ty in file_types:
        with open(src_file.format(ty)) as fp:
            cur_file = json.load(fp)
        for one in cur_file:
            set_id = one["set_id"]
            tot_set_id[ty].append(set_id)

            old_item_idx = [it["index"] for it in one["items"]]
            new_item_idx = list(range(1, len(old_item_idx) + 1))
            old_setid = map(lambda x: "{}_{}".format(*x), zip([set_id] * len(old_item_idx), old_item_idx))
            new_setid = map(lambda x: "{}_{}".format(*x), zip([set_id] * len(new_item_idx), new_item_idx))
            old_2_new_setid_mapping.update(zip(old_setid, new_setid))

    return old_2_new_setid_mapping


def collect_maryland_item_description():
    file_types = ['train', 'valid', 'test']
    fn_path = "../../label/{}_no_dup.json"

    all_sentences = set()
    for fyp in file_types:
        with open(fn_path.format(fyp)) as fp:
            fn = json.load(fp)
        all_sentences.update([it["name"] for one in fn for it in one["items"]])

    all_sentences.remove("")
    return list(all_sentences)


def convert_raw_file():
    all_sentences = collect_maryland_item_description()
    # old_2_new_setid_mapping = squeeze_benchmark_idx()
    with open(meta_path) as fp:
        meta_json = json.load(fp)

    for ty in file_types:
        ty_res = []
        with open(src_file.format(ty)) as fp:
            cur_file = json.load(fp)

        for one in cur_file:
            set_id = one["set_id"]
            converted = {"set_id": set_id}
            items = []
            if len(one["items"]) > 8:
                continue
            for it in one["items"]:
                item_id = it["item_id"]
                # new_setid, new_item_idx = old_2_new_setid_mapping["{}_{}".format(set_id, it["index"])].split("_")
                # assert new_setid == set_id
                items.append(
                    {"index": it["index"], "item_id": item_id, "categoryid": meta_json[item_id]["category_id"],
                     # "name": meta_json[item_id]['title'] or meta_json[item_id]['url_name']}
                     "name": random.choice(all_sentences)}
                )

            converted["items"] = items
            ty_res.append(converted)

        with open(output_file.format(ty), 'w') as fp:
            json.dump(ty_res, fp)


def convert_test_jobs():
    def rerange_question_sample(one):
        inner_setid=one["question"][0].split("_")[0]
        inner_blk_pos=one['blank_position']
        inner_ans=one["answers"]
        inner_right_setididx="{}_{}".format(inner_setid,inner_blk_pos)
        inner_ans.remove(inner_right_setididx)
        inner_ans.insert(0,inner_right_setididx)
        assert one["answers"][0]==inner_right_setididx
        assert len(one["answers"])==4
        return one



    # filter <=8
    with open(output_cp, 'w') as fp_cp:
        with open(compatibility_file) as fp:
            lines = fp.readlines()
            new_cnt = 0
            for line in lines:
                if len(line.strip().split()) < 10:
                    fp_cp.write(line)
                    new_cnt += 1

    print("cp: {} ==> {}".format(len(lines), new_cnt))

    with open(output_fitb, 'w') as fp_fitb:
        res_fitb = []
        with open(FITB_file) as fp:
            fitb = json.load(fp)
            for one in fitb:
                if len(one["question"]) < 8:
                    res_fitb.append(rerange_question_sample(one))

        json.dump(res_fitb, fp_fitb)

    print("fitb: {} ==> {}".format(len(fitb), len(res_fitb)))

    ##### retrieval #####
    all_sentences = collect_maryland_item_description()
    # old_2_new_setid_mapping = squeeze_benchmark_idx()
    with open(meta_path) as fp:
        meta_json = json.load(fp)

    for ty in ['test']:
        ty_res = []
        with open(src_file.format(ty)) as fp:
            cur_file = json.load(fp)

        for one in cur_file:
            set_id = one["set_id"]
            converted = {"set_id": set_id}
            items = []
            for it in one["items"]:
                item_id = it["item_id"]
                items.append(
                    {"index": it["index"], "item_id": item_id, "categoryid": meta_json[item_id]["category_id"],
                     "name": ""}
                )

            converted["items"] = items
            ty_res.append(converted)

        with open(output_retreival_features, 'w') as fp:
            json.dump(ty_res, fp)

    with open(output_retreival, 'w') as fp_retrieval:
        res_retrieval = []
        with open(retrieval_file) as fp:
            retr = json.load(fp)
            for one in retr:
                if len(one["question"]) < 8:
                    res_retrieval.append({
                        "question": one["question"],
                        "blank_position": int(one["right"].split("_")[-1]),
                        "answers": [one["right"]] + one["candidate"]
                    })

        json.dump(res_retrieval, fp_retrieval)

    print("retrieval: {} ==> {}".format(len(retr), len(res_retrieval)))


if __name__ == '__main__':
    # convert_raw_file()
    convert_test_jobs()
    # squeeze_benchmark_idx()
    # generate_benchmark_setid2imid()