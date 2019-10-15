import pickle
import argparse
import gzip
import json
import collections
import os
import re

import pickle

if __name__ == '__main__':

    all_sents = []
    for i in range(4):
        file ="/data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_{}.pk".format(i)
        all_sents.extend(pickle.load(open(file,"rb")))
    output_file = "/data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_all.json"
    with open(output_file, 'w', encoding='utf-8') as fout:
        for example in all_sents:
            fout.write(json.dumps(example) + '\n')
    print(len(all_sents))
    print(all_sents[0])
    print("Finised dump:", output_file)

    print("Total sents:",len(all_sents))
    examples_ids = []
    for e in all_sents:
        examples_ids.append(e["example_id"])
    examples_num = len(list(set(examples_ids)))
    print("Total nq_examples:",examples_num)
    print("Averaged:", len(all_sents)/examples_num)
