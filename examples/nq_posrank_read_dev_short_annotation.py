import pickle
import argparse
import gzip
import json
import collections
import glob
from multiprocessing import cpu_count
Thread_num = cpu_count()
print("Thread_num",Thread_num)
from multiprocessing import Pool
from functools import partial
import logging
logger = logging.getLogger(__name__)

'''
All train gzips:
read each line
and return a list about the short answer:
[eid,start_token,end_token,True/False]
Last one is whether has a short answer
'''

def read_annotation_for_devzip(input_file):
    train_annos_short = []
    print("Start:",input_file)
    # for input_file in input_files:
    with gzip.open(input_file, "r") as input_jsonl:
        for line in input_jsonl:
            e = json.loads(line, object_pairs_hook=collections.OrderedDict)
            eid = e["example_id"]
            for anno in e["annotations"]:
                short_ans = anno["short_answers"]
                if len(short_ans) == 0:
                    train_annos_short.append([eid,-1, -1, False])
                else:
                    answer = short_ans[0]
                    start_tok = answer["start_token"]
                    end_tok = answer["end_token"]
                    if start_tok == -1 or end_tok == -1:
                        train_annos_short.append( [eid,start_tok, end_tok, False])
                    else:
                        train_annos_short.append( [eid,start_tok, end_tok, True])
    print("End {} and total examples {}".format(input_file, len(train_annos_short)))
    return train_annos_short

def multiple_read(all_input_files):
    print('Multiprocessing!')
    features_initial = []
    size = int(len(all_input_files) / Thread_num)
    if size <1:
        size = 1
    with Pool(Thread_num) as p:
        annotate = partial(read_annotation_for_devzip)
        from tqdm import tqdm
        examples = list(tqdm(p.imap(annotate, all_input_files,chunksize=size),
                                     total=len(all_input_files),desc="reading gzips"))
    examples = [example for entry_examples in examples for example in entry_examples]
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gzip_dir", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    args = parser.parse_args()
    # -------------------------------------generated labels-------------------------------------------

    input_files = []
    for path in glob.glob("{}/*.gz".format(args.input_gzip_dir)):
        input_files.append(path)
    print("Total gzips:", len(input_files))

    annoted_short_labels = multiple_read(input_files)#[[eid,starttok,endtok,True/False(has answer)]]


    count_has_short =0
    count_no_short = 0
    for e in annoted_short_labels:
        if e[-1]:
            count_has_short+=1
        else:
            count_no_short+=1
    print("has short ans:",count_has_short)
    print("no short ans:",count_no_short)

    annoted_short_labels_dict = {}
    for e in annoted_short_labels:
        if e[0] in annoted_short_labels_dict:
            annoted_short_labels_dict[e[0]].append(e[1:])
        else:
            annoted_short_labels_dict[e[0]] = [e[1:]]
    pickle.dump(annoted_short_labels_dict, open(args.output_file, "wb"))
    print("total examples:",len(annoted_short_labels_dict))
    print("Dumpted annotations into {}".format(args.output_file))