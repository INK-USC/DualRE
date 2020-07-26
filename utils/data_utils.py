"""Preprocess data from both semeval and tacred"""
import os
import argparse
import json
import random
import collections
from pathlib import Path

random.seed(42)


def mask_tokens(tokens, subj, obj):
    (i1, j1, t1), (i2, j2, t2) = subj, obj
    tokens = tokens[:i1] + ["SUBJ-%s" % t1] * (j1 + 1 - i1) + tokens[j1 + 1 :]
    tokens = tokens[:i2] + ["OBJ-%s" % t2] * (j2 + 1 - i2) + tokens[j2 + 1 :]
    return tokens


def get_pst(tokens, i, j):
    pst = list(range(-i, 0)) + [0] * (j + 1 - i) + list(range(1, len(tokens) - j))
    return list(map(str, pst))


def convert_tacred_format(data_name, in_dir, out_dir):
    fname = in_dir / ("%s.json" % data_name)
    oname = out_dir / ("%s.json" % data_name)
    instances = []
    for data in json.load(open(fname)):
        if "token" in data:
            data["tokens"] = data.pop("token")
        data.pop("stanford_deprel", None)
        data.pop("stanford_head", None)
        data["subj_pst"] = get_pst(data["tokens"], data["subj_start"], data["subj_end"])
        data["obj_pst"] = get_pst(data["tokens"], data["obj_start"], data["obj_end"])
        data["tokens"] = mask_tokens(
            data["tokens"],
            (data["subj_start"], data["subj_end"], data["subj_type"]),
            (data["obj_start"], data["obj_end"], data["obj_type"]),
        )
        data["pr_confidence"] = 1
        data["sl_confidence"] = 1
        instances.append(data)
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    with open(oname, "w") as o:
        for instance in instances:
            o.write(json.dumps(instance) + "\n")
    return instances


def stratified_sample(data_dict, ratio):
    new_data, rest_data = [], []
    for _, data_list in data_dict.items():
        idxs = set(random.sample(range(len(data_list)), round(ratio * len(data_list))))
        new_data += [data_list[i] for i in idxs]
        rest_data += [data for i, data in enumerate(data_list) if i not in idxs]
    random.shuffle(new_data)
    random.shuffle(rest_data)
    return new_data, rest_data


def split_parts(fname, initial_train_ratio, onames):
    all_data = [json.loads(line) for line in open(fname, "r")]
    data_dict = collections.defaultdict(list)
    for entry in all_data:
        data_dict[entry["relation"]].append(entry)
    train_data, raw_data = stratified_sample(data_dict, initial_train_ratio)
    for entry in raw_data:
        entry["pr_confidence"] = 0
        entry["sl_confidence"] = 0
    for data, oname in zip((train_data, raw_data), onames):
        oname = fname.parent / oname
        print(oname)
        with open(oname, "w") as o:
            for d in data:
                o.write(json.dumps(d, ensure_ascii=True) + "\n")


def sample_from_data(fname, intial_ratio, absolute_ratios):
    data = [json.loads(line) for line in open(fname, "r")]
    data_dict = collections.defaultdict(list)
    for entry in data:
        data_dict[entry["relation"]].append(entry)
    for absolute_ratio in absolute_ratios:
        relative_ratio = absolute_ratio / intial_ratio
        new_data, _ = stratified_sample(data_dict, relative_ratio)
        oname = fname.parent / ("%s-%s.json" % (fname.stem.split("-")[0], absolute_ratio))
        print(oname)
        with open(oname, "w") as o:
            for d in new_data:
                o.write(json.dumps(d, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="tacred")
    parser.add_argument("--in_dir", type=str, default="", required=True)
    parser.add_argument("--out_dir", type=str, default="./json-new", required=True)
    args = vars(parser.parse_args())

    in_dir = Path(args["in_dir"])
    out_dir = Path(args["out_dir"])

    print("Reading from raw data...")
    for split in ["dev", "train", "test"]:
        if args["data_name"] == "tacred":
            convert_tacred_format(split, in_dir, out_dir)
        elif args["data_name"] == "semeval":
            pass
        else:
            raise ValueError("Data type %s not accepted." % args["data_name"])

    print("Splitting into train and raw...")
    initial_train_ratio = 0.5
    initial_raw_ratio = 1 - initial_train_ratio
    split_parts(
        out_dir / "train.json",
        initial_train_ratio,
        [f"train-{initial_train_ratio}.json", f"raw-{initial_raw_ratio}.json"],
    )
    print("Sample from data...")
    sample_from_data(out_dir / f"train-{initial_train_ratio}.json", initial_train_ratio, [0.05, 0.1, 0.3])
    sample_from_data(out_dir / f"raw-{initial_raw_ratio}.json", initial_train_ratio, [])


if __name__ == "__main__":
    main()
