# coding=utf-8
import pickle
import logging
import collections
import json
import re
import argparse

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
logger = logging.getLogger(__name__)

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

#--------------------------lqq added ---------------------------------------------------------
def write_nq_predictions(all_examples, all_features, all_results, n_best_size,
                        max_answer_length, do_lower_case, output_prediction_file,
                        verbose_logging,
                        version_2_with_negative,
                        nbest_pred_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    # all_predictions = collections.OrderedDict()
    all_nq_predictions = []
    # scores_diff_json = collections.OrderedDict()
    all_nq_nbest_predictions ={}
    from tqdm import tqdm
    for (example_index, example) in tqdm(enumerate(all_examples)):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if (start_index not in feature.token_to_orig_map) or (end_index not in feature.token_to_orig_map):
                        continue
                    else:
                        nq_s_idx = example.nq_context_map[feature.token_to_orig_map[start_index]]
                        nq_e_idx = example.nq_context_map[feature.token_to_orig_map[end_index]]
                        if nq_s_idx <0:
                            continue
                        if nq_e_idx <0:
                            continue
                        if nq_e_idx < nq_s_idx:
                            continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue

                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit","start_nq_idx","end_nq_idx"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)


                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

                nq_start = example.nq_context_map[orig_doc_start]  # lq_added
                nq_end = example.nq_context_map[orig_doc_end]  # lq_added
            else:
                final_text = ""
                seen_predictions[final_text] = True
                nq_start = -1  # lq_added
                nq_end = -1  # lq_added

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_nq_idx=nq_start,
                    end_nq_idx=nq_end))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        start_nq_idx=-1,
                        end_nq_idx=-1
                    ))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_nq_idx=-1, end_nq_idx=-1))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_nq_idx=-1, end_nq_idx=-1))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        # probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            # output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_short_idx"] = entry.start_nq_idx#lq added
            output["end_short_idx"] = entry.end_nq_idx#lq added
            output["score"] = entry.start_logit+entry.end_logit-score_null

            long_start = -1
            long_end = -1
            for (candidate_long_s, candidate_long_e) in example.nq_long_candidates:
                if candidate_long_s <= entry.start_nq_idx and candidate_long_e >= entry.end_nq_idx:
                    long_start = candidate_long_s
                    long_end = candidate_long_e
                    break
            output["start_long_idx"] = long_start
            output["end_long_idx"] = long_end
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_nq_nbest_predictions[example.qas_id] = nbest_json
        if version_2_with_negative:
            score_diff =  best_non_null_entry.start_logit+best_non_null_entry.end_logit -score_null#score is the cls.startlogits+cls.endlogits
            short_start = best_non_null_entry.start_nq_idx
            short_end = best_non_null_entry.end_nq_idx
            long_start = -1
            long_end = -1
            for (candidate_long_s,candidate_long_e) in example.nq_long_candidates:
                if candidate_long_s <= short_start and candidate_long_e >= short_end:
                    long_start = candidate_long_s
                    long_end = candidate_long_e
                    break

            all_nq_predictions.append({
                "example_id": int(example.qas_id),
                "long_answer": {
                    "start_token": long_start,
                    "end_token": long_end,
                    "start_byte": -1,
                    "end_byte": -1
                },
                "long_answer_score": score_diff,
                "short_answers": [{
                    "start_token": short_start,
                    "end_token": short_end+1,# pay attention here! if start == end, nq will return "a wrong span"!
                    "start_byte": -1,
                    "end_byte": -1
                }],
                "short_answers_score": score_diff,
                "yes_no_answer": "NONE"
            })

    predictions_json = {"predictions": all_nq_predictions}
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(predictions_json, indent=4) + "\n")

    pickle.dump(all_nq_nbest_predictions,open(nbest_pred_file,"wb"))
    logger.info("Pickle dumped nbest predictions to: %s" % (nbest_pk_file))

    return all_nq_predictions,all_nq_nbest_predictions
#--------------------------lqq end ------------------------------------------------------------
def convert_predwithsent_cls_format_train(all_pred_with_sent_data,example_labels):
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
    # data = pickle.load(open(predwithsent_files,"rb"))
    for (eid,ans_list) in all_pred_with_sent_data.items():
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
            if ana["sentence"] in sents_dict:
                continue
            else:
                sents_dict[ana["sentence"]] = [ana["sent_start_nq_idx"],ana["sent_end_nq_idx"]]

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



def npred_2_npredwithsent_Dev(all_nbest_predictions,all_examples_dict,example_labels):
    count_all_sents = 0
    global_id = 10000000000
    nbest_pred_with_sent = []
    count_has_answer = 0
    count_no_answer = 0
    sentlist = []
    from tqdm import tqdm
    for (eid, nbest_pred) in tqdm(all_nbest_predictions.items()):
        example = all_examples_dict[eid]
        # ------------nq_context_map to a dict---------------
        map_dict = {}
        for (i, n) in enumerate(example.nq_context_map):
            if n != -1:
                map_dict[n] = i
        # ------------sentence selector----------------------
        doc_tokens = example.doc_tokens
        idx = [i for i, n in enumerate(doc_tokens) if n == "." or n == "!" or n == "?"]
        if len(idx) == 0:
            idx = [len(doc_tokens)]
        # elif idx[-1] != len(doc_tokens):
        #     idx = idx + [len(doc_tokens)]
        sent_ends = list(map(lambda x: x + 1, idx))
        sent_starts = [0] + sent_ends
        sent_ends.append(len(doc_tokens) + 1)
        sent_spans = list(zip(sent_starts, sent_ends))  # print(" ".join(doc_tokens[start:end]))
        # ---------------------------------------------------

        for ans in nbest_pred:
            ans["sentence"] = ""

            if ans["start_short_idx"] == -1 or ans["end_short_idx"] == -1:
                # print("this is a null answer!")
                continue
            elif (ans["start_short_idx"] not in map_dict) or (ans["end_short_idx"] not in map_dict):
                print("the answer span is not right")
            else:
                start_tok_idx = map_dict[ans["start_short_idx"]]
                end_tok_idx = map_dict[ans["end_short_idx"]]

                start_of_sentence = 0
                end_of_sentence = 0
                for i in range(len(sent_starts) - 1):
                    if start_tok_idx >= sent_starts[i] and start_tok_idx < sent_starts[i + 1]:
                        start_of_sentence = sent_starts[i]
                        break
                if end_tok_idx < sent_ends[0]:
                    end_of_sentence = sent_ends[0]
                else:
                    for i in range(len(sent_ends) - 1):
                        if end_tok_idx >= sent_ends[i] and end_tok_idx < sent_ends[i + 1]:
                            end_of_sentence = sent_ends[i + 1]
                            break
                sentlist.append(" ".join(doc_tokens[start_of_sentence:end_of_sentence]))
                ans["sentence"] = " ".join(doc_tokens[start_of_sentence:end_of_sentence])
                ans["question"] = example.question_text
                ans["example_id"] = eid
                ans["label"] = "0"#no label
                if args.is_adddevlabel:#add label
                    if int(eid) in example_labels:
                        for one_label in example_labels[int(eid)]:
                            if one_label[-1] and one_label[0] >= start_of_sentence and one_label[1] <= end_of_sentence:
                                ans["label"] = '1'
                                count_has_answer += 1
                                break
                ans["id"] = global_id
                global_id+=1
                nbest_pred_with_sent.append(ans)
                if ans["sentence"] == "":
                    print("!")
                if ans["text"] not in ans["sentence"]:
                    print("answer: {}, sent: {}".format(ans["text"], ans["sentence"]))

        count_all_sents += len(set(sentlist))

    logger.info("Num of examples: %s" % (str(len(nbest_pred_with_sent))))
    logger.info(
        "Averaged distinguished sentences of each example: %s" % (str(count_all_sents / len(nbest_pred_with_sent))))
    pickle.dump(nbest_pred_with_sent, open(nbest_pred_with_sent_file, "wb"))
    logger.info("Dumped all_nbest_with_sents to : %s" % (nbest_pred_with_sent_file))
    logger.info("Labeled has answer: %d" % (count_has_answer))
    return nbest_pred_with_sent


def npred_2_npredwithsent_Train(all_nbest_predictions,all_examples_dict):
    all_nbest_pred_with_sents = {}
    count_all_sents = 0
    from tqdm import tqdm
    for (eid, nbest_pred) in tqdm(all_nbest_predictions.items()):
        example = all_examples_dict[eid]
        # ------------nq_context_map to a dict---------------
        map_dict = {}
        for (i, n) in enumerate(example.nq_context_map):
            if n != -1:
                map_dict[n] = i
        # ------------sentence selector----------------------
        doc_tokens = example.doc_tokens
        idx = [i for i, n in enumerate(doc_tokens) if n == "." or n == "!" or n == "?"]
        if len(idx) == 0:
            idx = [len(doc_tokens)]
        sent_ends = list(map(lambda x: x + 1, idx))
        sent_starts = [0] + sent_ends
        sent_ends.append(len(doc_tokens) + 1)
        sent_spans = list(zip(sent_starts, sent_ends))  # print(" ".join(doc_tokens[start:end]))
        # ---------------------------------------------------
        nbest_pred_with_sent = []
        sentlist = []
        for ans in nbest_pred:
            print(ans)

            if ans["start_nq_idx"] == -1 or ans["end_nq_idx"] == -1:
                # print("this is a null answer!")
                continue
            elif (ans["start_nq_idx"] not in map_dict) or (ans["end_nq_idx"] not in map_dict):
                print("the answer span is not right")
            else:
                start_tok_idx = map_dict[ans["start_nq_idx"]]
                end_tok_idx = map_dict[ans["end_nq_idx"]]

                start_of_sentence = 0
                end_of_sentence = 0
                for i in range(len(sent_starts) - 1):
                    if start_tok_idx >= sent_starts[i] and start_tok_idx < sent_starts[i + 1]:
                        start_of_sentence = sent_starts[i]
                        break
                if end_tok_idx < sent_ends[0]:
                    end_of_sentence = sent_ends[0]
                else:
                    for i in range(len(sent_ends) - 1):
                        if end_tok_idx >= sent_ends[i] and end_tok_idx < sent_ends[i + 1]:
                            end_of_sentence = sent_ends[i + 1]
                            break
                sentlist.append(" ".join(doc_tokens[start_of_sentence:end_of_sentence]))
                ans["sentence"] = " ".join(doc_tokens[start_of_sentence:end_of_sentence])
                ans["sent_start_doc_token_idx"] = start_of_sentence
                ans["sent_end_doc_token_idx"] = end_of_sentence - 2
                ans["sent_start_nq_idx"] = example.nq_context_map[start_of_sentence]
                ans["sent_end_nq_idx"] = example.nq_context_map[end_of_sentence - 2]
                ans["question"] = example.question_text
                ans["doc_tokens"] = example.doc_tokens
                ans["doc_tokens_nqidx_map"] = example.nq_context_map

                nbest_pred_with_sent.append(ans)
                if ans["sentence"] == "":
                    print("!")
                if ans["text"] not in ans["sentence"]:
                    print("answer: {}, sent: {}".format(ans["text"], ans["sentence"]))

        all_nbest_pred_with_sents[eid] = nbest_pred_with_sent
        count_all_sents += len(set(sentlist))

    logger.info("Num of examples: %s" % (str(len(all_nbest_pred_with_sents))))
    logger.info(
        "Averaged distinguished sentences of each example: %s" % (str(count_all_sents / len(all_nbest_predictions))))
    pickle.dump(all_nbest_pred_with_sents, open(nbest_pred_with_sent_file, "wb"))
    logger.info("Dumped all_nbest_with_sents to : %s" % (nbest_pred_with_sent_file))
    return all_nbest_pred_with_sents


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--example_pk_file", default=None, type=str, required=True)
    parser.add_argument("--feature_pk_file", default=None, type=str, required=True)
    parser.add_argument("--results_pk_file", default=None, type=str, required=True)

    parser.add_argument("--output_pred_file", default=None, type=str, required=True)
    parser.add_argument("--output_nbest_pk_file", default=None, type=str, required=True)
    parser.add_argument("--output_nbest_pred_with_sent_file", default=None, type=str, required=True)


    parser.add_argument("--train_anno_file", default=None, type=str)
    parser.add_argument("--is_training", action='store_true')
    parser.add_argument("--is_adddevlabel", action='store_true')
    parser.add_argument("--dev_anno_file", default=None, type=str)
    parser.add_argument("--output_cls_file", default=None, type=str, required=True)
    args = parser.parse_args()
    if args.is_training:
        assert args.train_anno_file != None
    if args.is_add_dev_label:
        assert args.dev_anno_file != None
    #--------------------------------------input files-----------------------------------------
    example_file = args.example_pk_file
    feature_file = args.feature_pk_file
    results_file = args.results_pk_file
    #-------------------------------------output files-----------------------------------------
    nbest_pk_file = args.output_nbest_pk_file# exampleid:{30 answer candidates}
    one_pred_json_file = args.output_pred_file#offical prediction file
    nbest_pred_with_sent_file = args.output_nbest_pred_with_sent_file#example:{30 answer candidates, and each answer is associated with its sentences}

    #-------------------------------------Load data-------------------------------------------
    all_examples = pickle.load(open(example_file, "rb"))
    all_features = pickle.load(open(feature_file, "rb"))
    all_results = pickle.load(open(results_file, "rb"))
    #
    all_examples_dict = {}
    for e in all_examples:
        all_examples_dict[e.qas_id] = e
    # #----------------------------step1: generated candidate answers--------------------
    output_prediction_file = one_pred_json_file
    all_nq_prediction, all_nbest_predictions = write_nq_predictions(all_examples, all_features, all_results,
                                                                    n_best_size=30,
                                                                    max_answer_length=30,
                                                                    do_lower_case=True,
                                                                    output_prediction_file=output_prediction_file,
                                                                    verbose_logging=False,
                                                                    version_2_with_negative=True,
                                                                    nbest_pred_file=nbest_pk_file)
    print("LQ1:",len(all_nq_prediction),len(all_nbest_predictions))
    if args.is_training:
        #--------------------step2:gent the sentence for each answer---------------
        nbest_pred_with_sents = npred_2_npredwithsent_Train(all_nbest_predictions,all_examples_dict)
        example_labels = pickle.load(open(args.train_anno_file,"rb"))#add label
        #-------------------step3: get the sentence label and convert to cls format--------
        train_examples_cls = convert_predwithsent_cls_format_train(nbest_pred_with_sents,example_labels)
        pickle.dump(train_examples_cls,open(args.output_cls_file,"wb"))
    else:
        # -----------------step2:gent the sentence for each answer and convert to cls format---------------
        if args.is_adddevlabel:
            example_labels = pickle.load(open(args.dev_anno_file, "rb"))
        else:
            example_labels=None
        nbest_pred_with_sents_cls = npred_2_npredwithsent_Dev(all_nbest_predictions,all_examples_dict,example_labels)
        pickle.dump(nbest_pred_with_sents_cls, open(args.output_cls_file, "wb"))
        print("Dumped npredwitsent_cls format to ",args.output_cls_file)

        #---------------add label---for research---------------------------------------------

        # print("sents:",len(nbest_pred_with_sents_cls))
        # examples_id = []
        # for e in nbest_pred_with_sents_cls:
        #     examples_id.append(e["example_id"])
        # print("examples:",len(list(set(examples_id))))

