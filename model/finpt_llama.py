#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import List, Optional, Tuple, Union
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LLAMA_START_DOCSTRING
from transformers.models.t5.modeling_t5 import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers import LlamaModel, LlamaPreTrainedModel


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class FinptLlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"classifier.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)

        self.dropout_prob = 0.1
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.tokenizer = None
        self.neg_to_pos = float(1.0)
        self.use_pos_weight = False
        self.nan_batch_count = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # logits = self.classifier(hidden_states)

        if np.isnan(hidden_states.cpu().detach().numpy()).sum() > 0:
            self.nan_batch_count += 1

        # if input_ids is not None:
        #     batch_size = input_ids.shape[0]
        # else:
        #     batch_size = inputs_embeds.shape[0]
        #
        # if self.tokenizer.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.tokenizer.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         sequence_lengths = (torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1).to(logits.device)
        #     else:
        #         sequence_lengths = -1
        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        # use the hidden states of the last token only
        batch_size = attention_mask.size(0)
        sequence_lengths = attention_mask.sum(-1)
        cls_hidden = []
        for b_idx in range(batch_size):
            cur_seq_len = sequence_lengths[b_idx]
            cls_hidden.append(hidden_states[b_idx: b_idx + 1, cur_seq_len - 1, :])
        cls_hidden = torch.cat(cls_hidden, dim=0)

        cls_hidden = self.dropout(cls_hidden)
        cls_logits = self.classifier(cls_hidden)
        pooled_logits = cls_logits

        loss = None
        if labels is not None:
            labels = labels.to(cls_logits.device)
            if self.config.problem_type is None:
                assert isinstance(self.num_labels, int) and self.num_labels >= 1
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)

            elif self.config.problem_type == "single_label_classification":
                if self.use_pos_weight:
                    cur_dev = cls_logits.device

                    pos_weight = torch.tensor([1.0, self.neg_to_pos], dtype=torch.float32, device=cur_dev)
                    loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                    labels_bce = F.one_hot(labels, num_classes=self.num_labels).to(dtype=torch.float32, device=cur_dev)
                    pooled_logits_bce = pooled_logits.to(dtype=torch.float32, device=cur_dev)
                    loss = loss_fct(pooled_logits_bce, labels_bce)
                    loss = loss.to(dtype=cls_logits.dtype, device=cur_dev)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
