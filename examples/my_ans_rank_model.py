from transformers.modeling_bert import *
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
class BertForAnsRank(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForAnsRank, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.weight_linear = nn.Linear(config.hidden_size,1,bias=False)
        self.vec_linear = nn.Linear(config.hidden_size,config.hidden_size,bias=False)
        self.score_linear = nn.Linear(config.hidden_size,1,bias=False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                hard_label=None, soft_label=None,ans_span=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]#batch,seq_len,hidden_size
        batch_size = sequence_output.shape[0]
        sequence_len = sequence_output.shape[1]
        hidden_size = sequence_output.shape[2]
        ans_size = ans_span.shape[1]

        #--------------------------------------------------------------------
        sequence_output = torch.unsqueeze(sequence_output, 1)
        sequence_output = sequence_output.repeat(1, ans_size, 1, 1)#batch,ans_size,seq_len,hid_size
        ans_span = torch.unsqueeze(ans_span.float(),-1)#batch,ans_size,hid_size,1
        ans_repre = sequence_output.mul(ans_span)#batch,ans_size,seq_len,hid_size

        word_wight_yita = self.softmax(self.weight_linear(ans_repre)).transpose(-1,-2)#batch,ans_size,1,seq_len

        ans_repre = ans_repre.view(-1, ans_repre.shape[-2], ans_repre.shape[-1])#batch*ans_size,seq_len,hid_size
        word_wight_yita = word_wight_yita.view(-1, word_wight_yita.shape[-2], word_wight_yita.shape[-1])#batch*ans_size,1,seq_len

        ans_repre_weightedsum = torch.bmm(word_wight_yita, ans_repre).view(batch_size, ans_size, -1)#batch,ans_size,hid_size

        final_ans_scores = self.score_linear(torch.tanh(self.vec_linear(ans_repre_weightedsum)))#b,ans_size
        #self.relu(self.vec_linear(ans_repre_weightedsum))
        score_logits = final_ans_scores.squeeze(-1)#batch,ans_size

        outputs = (score_logits,)
        if hard_label is not None and soft_label is not None:
            # # If we are on multi-GPU, split add a dimension
            # if len(hard_label.size()) > 1:
            #     hard_label = hard_label.squeeze(-1)
            # if len(soft_label.size()) > 1:
            #     soft_label = soft_label.squeeze(-1)
            # # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # ignored_index = score_logits.size(1)
            # hard_label.clamp_(0, ignored_index)
            # soft_label.clamp_(0, ignored_index)

            hard_loss_fct = BCEWithLogitsLoss()#ignore_index=ignored_index
            hard_loss = hard_loss_fct(score_logits, hard_label.float())


            soft_loss_fct = MSELoss()
            soft_loss = soft_loss_fct(self.softmax(score_logits),soft_label)

            total_loss = (hard_loss + soft_loss) / 2
            outputs = (total_loss,) + outputs


        return outputs  # (loss), score_logits
