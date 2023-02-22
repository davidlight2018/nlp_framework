import torch
import torch.nn as nn
from typing import List, Optional
from transformers import BertModel, BertPreTrainedModel, AutoModel
from .layers.crf import CRF
# from torchcrf import CRF


class BertCrfForNer(BertModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size//2, bidirectional=True, batch_first=True,
                              num_layers=2, dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # biLSTM
        # sequence_output, hc = self.bilstm(sequence_output)

        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            # with CRF
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,)+outputs

            # without CRF
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs

        return outputs  # (loss), scores
