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

    """
    227705
OrderedDict([('text', 'J.J. Thomson'), ('start_logit', 2.622042655944824), ('end_logit', 3.1629788875579834), ('start_short_idx', 2157), ('end_short_idx', 2158), ('start_long_idx', 2094), ('end_long_idx', 2342), ('sentence', 'This was first demonstrated by J.J. Thomson in 1897 when , using a cathode ray tube , he found that an electrical charge would travel across a vacuum ( which would possess infinite resistance in classical theory ) .'), ('question', 'who proposed that electrons behave like waves and particles'), ('example_id', '-5501481664893105662'), ('id', 10000000000)])
Finised dump: /data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_all.json
Total sents: 227705
Total nq_examples: 7830
Averaged: 29.08109833971903
"""
