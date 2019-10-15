#!/bin/bash
for ((i=0;i<4;i++));do
{
python ./examples/nq_posrank_sentences_generation.py \
--example_pk_file /data/nieping/pytorch-transformers/models/wwm_test/examples_${i}_.pk \
--feature_pk_file /data/nieping/pytorch-transformers/models/wwm_test/features_${i}_.pk \
--results_pk_file /data/nieping/pytorch-transformers/models/wwm_test/allresults_${i}_.pk \
--output_nbest_pk_file /data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_pred_nbest{i}.pk \
--output_pred_file /data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_pred_${i}.json \
--output_nbest_pred_with_sent_file /data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_nbest_predwithsent_${i}.pk \
--output_cls_file /data/nieping/pytorch-transformers/data/nq_sentence_selector/dev_all/dev_predwithsent_cls_${i}.json

echo Done ${i}
} &
done
#finished