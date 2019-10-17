import json
import sys
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm
if __name__ == '__main__':
    # tok_th = float(sys.argv[1])
    # sent_th = float(sys.argv[2])
    cls_file = "../dev_eval_results.json"
    predwithsent_file = "../dev_predwithsent_cls_all.json"
    output_tok_file = "../orig_preds.json"
    output_sent_span_file = "../grads/pos_reranker_preds.json"
    # output_sent_span_file = "../grads/pos_reranker_preds_{}_{}.json".format(tok_th,sent_th)


    with open(cls_file) as fin:
        lines = fin.readlines()
        for line in lines:
            cls_preds = json.loads(line)
    sent_cls_predlabels = cls_preds["preds"]
    sent_cls_score = []
    for pred in cls_preds["logits"]:
        sent_cls_score.append(pred[1])
    sent_cls_scores_softmax = softmax(sent_cls_score)
    sent_cls_scores_norm = normalize(sent_cls_score)

    all_sents = []
    with open(predwithsent_file) as fin:
        lines = fin.readlines()
        for line in lines:
            all_sents.append(json.loads(line))
    tokspan_scores = []
    for i in range(len(all_sents)):
        tokspan_scores.append(all_sents[i]["score"])
    tokspan_scores_softmax = softmax(tokspan_scores)
    tokspan_scores_norm = normalize(tokspan_scores)


    example_dict = {}
    total_logits = []
    for i in range(len(sent_cls_scores_softmax)):
        sent = all_sents[i]
        sent["sent_cls_score_softmax"]=sent_cls_scores_softmax[i]
        sent["sent_cls_score_norm"] = sent_cls_scores_norm[i]
        sent["sent_cls_score"] = sent_cls_score[i]
        sent["score_norm"] = tokspan_scores_norm[i]
        sent["score_softmax"] = tokspan_scores_softmax[i]

        eid = sent["example_id"]
        if eid in example_dict:
            example_dict[eid].append(sent)
        else:
            example_dict[eid] = [sent]

    # print(len(example_dict))
    orig_preds = []
    reranker_preds = []
    count = 0
    changed = 0
    for (eid,ans_list) in example_dict.items():
        ans_list.sort(key=lambda x: (-x["score"]))
        orig_top = ans_list[0]
        orig_preds.append({
            "example_id": int(eid),
            "long_answer": {
                "start_token": orig_top["start_long_idx"],
                "end_token": orig_top["end_long_idx"],
                "start_byte": -1,
                "end_byte": -1
            },
            "long_answer_score": orig_top["score"],
            "short_answers": [{
                "start_token": orig_top["start_short_idx"],
                "end_token": orig_top["end_short_idx"] + 1,
                # pay attention here! if start == end, nq will return "a wrong span"!
                "start_byte": -1,
                "end_byte": -1
            }],
            "short_answers_score": orig_top["score"],
            "yes_no_answer": "NONE"
        })

        #-------------------------reranker--------------------------------

        ans_list.sort(key=lambda x: (-(x["sent_cls_score_norm"]+x["score_norm"])))
        reranker_top = ans_list[0]

        reranker_preds.append({
            "example_id": int(eid),
            "long_answer": {
                "start_token": reranker_top["start_long_idx"],
                "end_token": reranker_top["end_long_idx"],
                "start_byte": -1,
                "end_byte": -1
            },
            "long_answer_score":reranker_top["sent_cls_score_norm"]+reranker_top["score_norm"],
            "short_answers": [{
                "start_token": reranker_top["start_short_idx"],
                "end_token": reranker_top["end_short_idx"] + 1,
                # pay attention here! if start == end, nq will return "a wrong span"!
                "start_byte": -1,
                "end_byte": -1
            }],
            "short_answers_score": reranker_top["sent_cls_score_norm"]+reranker_top["score_norm"],
            "yes_no_answer": "NONE"
        })


    span_predictions_json = {"predictions": orig_preds}
    with open(output_tok_file, "w") as writer:
        writer.write(json.dumps(span_predictions_json, indent=4) + "\n")
    # print(len(reranker_preds))
    sent_predictions_json = {"predictions": reranker_preds}
    with open(output_sent_span_file, "w") as writer:
        writer.write(json.dumps(sent_predictions_json, indent=4) + "\n")
