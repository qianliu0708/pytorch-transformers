import pickle
import argparse
import gzip
import json
import collections
import os
import re
import glob
from multiprocessing import cpu_count
Thread_num = cpu_count()
print("Thread_num",Thread_num)
import multiprocessing
from multiprocessing import Pool
from functools import partial
import logging
logger = logging.getLogger(__name__)
def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "r")
    else:
        print("wrong file")
        exit()

def read_annotation_for_traingzip(input_files):
    train_annos_short = []
    print("start:",input_files)
    from tqdm import tqdm
    for input_file in tqdm(input_files):
        with _open(input_file) as input_jsonl:
            for line in input_jsonl:
                e = json.loads(line, object_pairs_hook=collections.OrderedDict)
                eid = e["example_id"]
                anno = e["annotations"][0]
                question = " ".join(e["question_tokens"])
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
    # print("total examples", len(example_short_anno))
    return train_annos_short

def multiple_read(all_input_files):
    logger.info('Multiprocessing!')
    features_initial = []
    with Pool(Thread_num) as p:
        annotate = partial(read_annotation_for_traingzip)

        piece_num = int(len(all_input_files) / Thread_num)
        example_chunks = [all_input_files[start:start + piece_num] for start in range(0, len(all_input_files), piece_num)]
        logger.info('total chunks: {}'.format(len(example_chunks)))

        for chunk_id, examples_part in enumerate(example_chunks):
            from tqdm import tqdm
            features_partial = list(tqdm(p.imap(annotate, examples_part),total=len(examples_part),desc="reading gzips"))
            features_initial.extend(features_partial)
            logger.info('processing chunk {}'.format(chunk_id))
    return features_initial




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
    pickle.dump(annoted_short_labels, open(args.output_file, "wb"))
    print("Dumpted annotations into {}".format(args.output_file))
    count_has_short =0
    count_no_short = 0
    for e in annoted_short_labels:
        if e[-1]:
            count_has_short+=1
        else:
            count_no_short+=1
    print("has short ans:",count_has_short)
    print("no short ans:",count_no_short)

