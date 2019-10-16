import json
if __name__ == '__main__':
    file = "../dev_eval_results.json"
    preds = "../dev_predwithsent_cls_all.json"
    with open(file) as fin:
        lines = fin.readlines()
        for line in lines:
            cls_preds = json.loads(line)
    all_sents = []
    with open(preds) as fin:
        lines = fin.readlines()
        for line in lines:
            all_sents.append(json.loads(line))

