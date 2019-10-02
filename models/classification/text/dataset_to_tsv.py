# this script is used to convert dataset to TSV
# output of this script would be
# 1. train.tsv {label, text}
# 2. test.tsv {label, text}
# 3. classes.txt {id, label}

import os
import argparse

from random import shuffle
from collections import defaultdict
from pre_process import cleanup_text

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="directory where we have data set in form label > file")
parser.add_argument("--test_ratio", default=0.20, help="(default: 0.20) ratio of sample for test set")
args = parser.parse_args()

if __name__ == "__main__":
    all_labels = [label for label in os.listdir(args.dataset)]
    label_to_file_index = defaultdict(list)

    with open("classes.txt", "w") as output:
        output.write("\n".join(all_labels))

    for label in os.listdir(args.dataset):
        for current in os.listdir(os.path.join(args.dataset, label)):
            label_to_file_index[label].append(os.path.join(args.dataset, label, current))

    train = open("train.tsv", "w")
    test = open("test.tsv", "w")

    for label in label_to_file_index:
        sample = label_to_file_index[label]
        shuffle(sample)
        for i in range(len(sample)):
            rec = "%s\t%s\n" % (all_labels.index(label) + 1, cleanup_text(open(sample[i], encoding="utf-8", errors="surrogateescape").read()))
            if i < int(args.test_ratio * len(sample)):
                test.write(rec)
            else:
                train.write(rec)

    print("Created: train.tsv, test.tsv and classes.txt")