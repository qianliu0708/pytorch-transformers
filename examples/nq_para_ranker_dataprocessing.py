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

def read_para_for_trainzip(input_file):
    train_annos_short = []
    print("Start:",input_file)
    # for input_file in input_files:
    with gzip.open(input_file, "r") as input_jsonl:
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
    print("End {} and total examples {}".format(input_file, len(train_annos_short)))
    return train_annos_short


def multiple_read(all_input_files):
    print('Multiprocessing!')
    features_initial = []
    size = int(len(all_input_files) / Thread_num)
    with Pool(Thread_num) as p:
        read_para = partial(read_para_for_trainzip)
        from tqdm import tqdm
        examples = list(tqdm(p.imap(read_para, all_input_files,chunksize=size),
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

    annoted_short_labels = multiple_read(input_files)  # [[eid,starttok,endtok,True/False(has answer)]]