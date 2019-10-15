import pickle
import argparse
import gzip
import json
import collections
import os
import re

def convert_predwithsent_cls_format(example_labels,predwithsent_files):
    '''

    :param example_labels: a dict{example_id:[start_nq_idx, end_nq_idx, False/True,question]}
    :param predwithsent_files: a dict {example_id:{generated from nq_posrank_Sentences_generation}}
    :return:
    '''
    all_sents_examples = []
    count_no_annotation_examples = 0
    count_has_annotation_examples = 0
    count_has_answer = 0
    count_no_answer = 0
    data = pickle.load(open(predwithsent_files,"rb"))
    for (eid,ans_list) in data.items():
        question = ans_list[0]["question"]
        #----------generated distinguished sentens
        sents_dict = {}
        if int(eid) in example_labels:
            label = example_labels[int(eid)]
            count_has_annotation_examples+=1
        else:
            print("an example without annotation")
            count_no_annotation_examples +=1
            continue

        for ana in ans_list:
            if ana["sent"] in sents_dict:
                continue
            else:
                sents_dict[ana["sent"]] = [ana["sent_start_nq_idx"],ana["sent_end_nq_idx"]]

        for (sent,span) in sents_dict.items():
            if label[-1] and label[0] >= span[0] and label[1] <= span[1]:
                sent_label = '1'
                count_has_answer+=1
            else:
                sent_label = '0'
                count_no_answer+=1
            all_sents_examples.append({
                'id':int(eid),
                'question': question,
                'sentence': sent,
                'sent_span': span,
                'label':sent_label,
            })
    print("no annotation examples:",count_no_annotation_examples)
    print("has annotation examples:",count_has_annotation_examples)
    print("All sents:",len(all_sents_examples))
    print("\t positive sent:",count_has_answer)
    print("\t negative sent:",count_no_answer)
    return all_sents_examples

import pickle

if __name__ == '__main__':

    all_sents = []
    for i in range(4):
        file ="/data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_{}.json".format(i)
        all_sents.extend(pickle.load(open(file,"rb")))
    output_file = "/data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_all.json"
    with open(output_file, 'w', encoding='utf-8') as fout:
        for example in all_sents:
            fout.write(json.dumps(example) + '\n')
    print(len(all_sents))
    print(all_sents[0])
    print("Finised dump:", output_file)