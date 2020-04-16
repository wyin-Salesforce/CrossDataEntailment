
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel, RobertaConfig, RobertaForSequenceClassification#, RobertaClassificationHead
from transformers.modeling_bert import BertPreTrainedModel

from bert_common_functions import store_transformers_models, get_a_random_batch_from_dataloader, cosine_rowwise_two_matrices

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
}


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def get_MNLI_as_train(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples_entail=[]
        examples_neutral=[]
        examples_contra=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "train-"+str(line_co-1)
                text_a = line[8].strip()
                text_b = line[9].strip()
                label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                if label == 'entailment':
                    examples_entail.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                elif label == 'neutral':
                    examples_neutral.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    examples_contra.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
            if line_co > 2000:
                break
        readfile.close()
        print('loaded  size:', line_co)
        return examples_entail, examples_neutral, examples_contra




    def get_RTE_as_train(self, filename, K, sampling_seed):
        '''first load all into lists'''
        readfile = codecs.open(filename, 'r', 'utf-8')
        entail_list = []
        not_entail_list = []
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                premise = line[1].strip()
                hypothesis = line[2].strip()
                label = line[3].strip()
                if label == 'entailment':
                    entail_list.append((premise, hypothesis))
                else:
                    not_entail_list.append((premise, hypothesis))
            line_co+=1
        readfile.close()
        print('RTE pos: ', len(entail_list), ' neg size: ', len(not_entail_list))
        #RTE pos:  1249  neg size:  1241

        '''now randomly sampling'''
        entail_size = len(entail_list)
        not_entail_size = len(not_entail_list)
        print('entail_size:', entail_size, 'not_entail_size:', not_entail_size)

        K=min(K, entail_size)
        if K == entail_size:
            sampled_entail = entail_list
        else:
            sampled_entail = random.Random(sampling_seed).sample(entail_list, K)

        if K <= not_entail_size:
            sampled_not_entail = random.Random(sampling_seed).sample(not_entail_list, K)
        else:
            sampled_not_entail = not_entail_list+random.Random(sampling_seed).sample(not_entail_list, K-not_entail_size)

        # print('sampled_entail size:', len(sampled_entail))
        # print('sampled_not_entail size:', len(sampled_not_entail))
        examples_entail=[]
        examples_neutral=[]
        examples_contra=[]

        for idd, pair in enumerate(sampled_entail):
            examples_entail.append(
                InputExample(guid='entail_'+str(idd), text_a=pair[0], text_b=pair[1], label='entailment'))

        # for idd, pair in enumerate(sampled_not_entail):
        #     if idd < int(K/2):
        #         '''neutral'''
        #         examples_neutral.append(
        #             InputExample(guid='neutral_'+str(idd), text_a=pair[0], text_b=pair[1], label='neutral'))
        #     else:
        #         '''contradiction'''
        #         examples_contra.append(
        #             InputExample(guid='contra_'+str(idd), text_a=pair[0], text_b=pair[1], label='contradiction'))
        # return examples_entail, examples_neutral+examples_neutral, examples_contra+examples_contra

        for idd, pair in enumerate(sampled_not_entail):
            '''neutral'''
            examples_neutral.append(
                InputExample(guid='neutral_'+str(idd), text_a=pair[0], text_b=pair[1], label='neutral'))
            '''contradiction'''
            examples_contra.append(
                InputExample(guid='contra_'+str(idd), text_a=pair[0], text_b=pair[1], label='contradiction'))
        return examples_entail, examples_neutral, examples_contra

    def get_RTE_as_dev(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "dev-"+str(line_co-1)
                text_a = line[1].strip()
                text_b = line[2].strip()
                # label = line[3].strip() #["entailment", "not_entailment"]
                label = 'entailment'  if line[3] == 'entailment' else 'neutral'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
            # if line_co > 20000:
            #     break
        readfile.close()
        print('loaded  size:', line_co-1)
        return examples

    def get_RTE_as_test(self, filename):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==3:
                guid = "test-"+str(line_co)
                text_a = line[1]
                text_b = line[2]
                '''for RTE, we currently only choose randomly two labels in the set, in prediction we then decide the predicted labels'''
                label = 'entailment'  if line[0] == '1' else 'neutral'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1

        readfile.close()
        print('loaded test size:', line_co)
        return examples


    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class Encoder(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.num_labels = config.num_labels
        '''this self.roberta can help the self.classifier to be initilaized'''
        self.roberta = None#RobertaModel(config)
        '''classifier for target domain'''
        self.classifier = RobertaClassificationHead(config)
        self.classifier_3_layers = RobertaClassificationHead_3_layers(config)
        self.classifier_3_layers.load_state_dict(self.classifier.state_dict())

        '''nearest neighbor parameters'''
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp_1 = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.mlp_2 = nn.Linear(config.hidden_size, 1, bias=False)

    def NearestNeighbor(self, sample_reps, sample_logits, query_reps, query_labels, mode='train_NN', loss_fct = None):
        '''
        mode: train_NN, train_CL, test
        '''
        sample_size = sample_reps.shape[0]
        query_size = query_reps.shape[0]
        hidden_size = query_reps.shape[1]

        # print('sample_size:', sample_size, 'query_size:', query_size, 'hidden_size:', hidden_size)


        repeat_sample_rep = torch.cat([sample_reps]*query_size, dim=0) #(9*batch_size, hidden)
        # print('repeat_sample_rep shape:', repeat_sample_rep.shape)
        # print('query_reps:', query_reps.shape)
        repeat_query_rep = query_reps.repeat(1, sample_size).view(-1, hidden_size)#(9*batch_size, hidden)

        '''? add similarity or something similar?'''
        mlp_input = torch.cat([
        repeat_query_rep, repeat_sample_rep,
        # repeat_batch_outputs - repeat_sample_rep,
        # cosine_rowwise_two_matrices(repeat_batch_outputs, repeat_sample_rep),
        repeat_query_rep*repeat_sample_rep
        ], dim=1) #(batch*class_size, hidden*2)
        '''??? add drop out here'''
        group_scores = torch.tanh(self.mlp_2(self.dropout(torch.tanh(self.mlp_1(self.dropout(mlp_input))))))#(batch*class_size, 1)
        group_scores_with_simi = group_scores + cosine_rowwise_two_matrices(repeat_query_rep, repeat_sample_rep)
        # group_scores = torch.tanh(self.mlp_2((torch.tanh(mlp_input))))#(9*batch_size, 1)
        # print('group_scores:',group_scores)

        similarity_matrix = group_scores_with_simi.reshape(query_size, sample_size)
        '''???note that the softmax will make the resulting logits smaller than LR'''
        query_logits_from_NN = torch.mm(nn.Softmax(dim=1)(similarity_matrix), sample_logits) #(batch, 3)
        if mode == 'test':
            return query_logits_from_NN
        else:
            loss_i = loss_fct(query_logits_from_NN.view(-1, self.num_labels), query_labels.view(-1))
            return loss_i



    def forward(self, target_sample_reps_logits_labels, target_sample_last3_reps, source_sample_reps_logits, source_batch_reps_labels,
                test_batch_reps_logits_labels, test_batch_last3_reps, source_reps_logits_history, target_reps_logits_history,
                mode='train_NN', loss_fct = None):
        '''
        mode: train_NN, train_CL, test
        '''
        '''input for training'''
        if mode == 'train_CL':
            target_sample_reps, target_sample_logits, target_sample_labels = target_sample_reps_logits_labels
        if mode == 'train_NN':
            target_sample_reps, target_sample_logits, target_sample_labels = target_sample_reps_logits_labels
            source_sample_reps, source_sample_logits = source_sample_reps_logits
            source_batch_reps, source_batch_labels = source_batch_reps_labels
        '''input for testing'''
        if mode == 'test':
            test_batch_reps, test_batch_logits, test_batch_labels = test_batch_reps_logits_labels
            source_sample_reps_history, source_sample_logits_history = source_reps_logits_history
            target_sample_reps_history, target_sample_logits_history = target_reps_logits_history


        if mode == 'train_NN':
            loss_target_pred_source = self.NearestNeighbor(target_sample_reps, target_sample_logits, source_batch_reps, source_batch_labels, mode='train_NN', loss_fct = loss_fct)
            loss_source_pred_source = self.NearestNeighbor(source_sample_reps, source_sample_logits, source_batch_reps, source_batch_labels, mode='train_NN', loss_fct = loss_fct)
            loss_source_pred_target = self.NearestNeighbor(source_sample_reps, source_sample_logits, target_sample_reps, target_sample_labels, mode='train_NN', loss_fct = loss_fct)

            NN_loss = loss_target_pred_source+loss_source_pred_source+loss_source_pred_target
            return NN_loss
        elif mode == 'train_CL':
            # four_layer_reps = torch.cat([target_sample_reps, target_sample_last3_reps], dim=1)
            # four_layer_reps = target_sample_reps + target_sample_last3_reps
            target_sample_CL_logits = self.classifier(target_sample_reps)
            target_sample_CL_logits_3_layers = self.classifier_3_layers(target_sample_last3_reps)
            # target_sample_CL_logits_3_layers = self.classifier(four_layer_reps)
            four_layers_logits = target_sample_CL_logits_3_layers+target_sample_CL_logits #target_sample_CL_logits+
            CL_loss = loss_fct(four_layers_logits.view(-1, self.num_labels), target_sample_labels.view(-1))
            return CL_loss
        else:
            '''testing'''
            '''first, get logits from RobertaForSequenceClassification'''
            logits_from_pretrained = test_batch_logits
            '''second, get logits from NN, two parts, one from source. one from target samples'''
            NN_logits_from_source = self.NearestNeighbor(source_sample_reps_history, source_sample_logits_history, test_batch_reps, None, mode='test', loss_fct = loss_fct)
            NN_logits_from_target = self.NearestNeighbor(target_sample_reps_history, target_sample_logits_history, test_batch_reps, None, mode='test', loss_fct = loss_fct)
            NN_logits_combine = NN_logits_from_source+NN_logits_from_target
            '''third, get logits from classification of the target domain'''
            CL_logits_from_target = (
                                    self.classifier(test_batch_reps)+
                                    self.classifier_3_layers(test_batch_last3_reps)
                                    # self.classifier(test_batch_reps + test_batch_last3_reps)
                                    )
            # print('logits_from_pretrained:', logits_from_pretrained)
            # print('NN_logits_combine:', NN_logits_combine)
            # print('CL_logits_from_target:', CL_logits_from_target)
            overall_test_batch_logits = logits_from_pretrained+NN_logits_combine+CL_logits_from_target
            # overall_test_batch_logits = logits_from_pretrained
            # overall_test_batch_logits = NN_logits_combine
            # overall_test_batch_logits = CL_logits_from_target#logits_from_pretrained+CL_logits_from_target
            pred_labels_batch = overall_test_batch_logits.argmax(dim=1)#torch.softmax(overall_test_batch_logits.view(-1, self.num_labels), dim=1).argmax(dim=1)
            pred_labels_batch_NN = NN_logits_combine.argmax(dim=1)
            pred_labels_batch_pre = logits_from_pretrained.argmax(dim=1)
            pred_labels_CL = CL_logits_from_target.argmax(dim=1)
            '''majority voting'''
            # pred_labels_batch_roberta = torch.softmax(logits_from_pretrained.view(-1, self.num_labels), dim=1).argmax(dim=1, keepdim=True) #(batch, 1)
            # pred_labels_batch_NN = torch.softmax(NN_logits_combine.view(-1, self.num_labels), dim=1).argmax(dim=1, keepdim=True)
            # pred_labels_batch_CL = torch.softmax(CL_logits_from_target.view(-1, self.num_labels), dim=1).argmax(dim=1, keepdim=True)
            # combine_pred_labels_batch = torch.cat([pred_labels_batch_roberta,pred_labels_batch_NN , pred_labels_batch_CL], dim=1) #(batch, 3)
            # pred_labels_batch = (combine_pred_labels_batch==0).sum(dim=1)
            # pred_labels_batch = 1-(pred_labels_batch>1).type(torch.cuda.LongTensor)#[entail_size_batch>1]=0



            return pred_labels_batch, pred_labels_batch_NN, pred_labels_batch_pre, pred_labels_CL




class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationHead_3_layers(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead_3_layers, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)

        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # x = self.dense2(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        #
        # x = self.dense3(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        x = self.out_proj(x)
        return x

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--NN_iter_limit',
                        type=int,
                        default=100,
                        help="random seed for initialization")
    parser.add_argument('--k_shot',
                        type=int,
                        default=3,
                        help="random seed for initialization")
    parser.add_argument('--sampling_seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--stilts_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--NN_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels() #["entailment", "neutral", "contradiction"]
    num_labels = len(label_list)



    source_examples_entail, source_examples_neutral, source_examples_contra = processor.get_MNLI_as_train('/export/home/Dataset/glue_data/MNLI/train.tsv') #train_pu_half_v1.txt
    target_samples_entail, target_samples_neutral, target_samples_contra = processor.get_RTE_as_train('/export/home/Dataset/glue_data/RTE/train.tsv', args.k_shot, args.sampling_seed)

    pretrain_model_dir = '/export/home/Dataset/BERT_pretrained_mine/crossdataentail/trainMNLItestRTE/0.8664259927797834-0.8106035345115038'
    roberta_seq_model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels, output_hidden_states=True)
    roberta_seq_model.to(device)
    roberta_seq_model.eval()
    # roberta_seq_model.config.output_hidden_states=True
    # print('roberta_seq_model.config:', roberta_seq_model.config)
    # exit(0)

    model = Encoder.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)
    # store_bert_model(model, tokenizer.vocab, '/export/home/workspace/CrossDataEntailment/models', 'try')
    # exit(0)
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0
    if args.do_train:
        source_features = convert_examples_to_features(
            source_examples_entail + source_examples_neutral + source_examples_contra,
            label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        source_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
        source_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
        source_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
        source_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)

        source_data = TensorDataset(source_all_input_ids, source_all_input_mask, source_all_segment_ids, source_all_label_ids)
        source_sampler = RandomSampler(source_data)
        source_samples_dataloader = DataLoader(source_data, sampler=source_sampler, batch_size=9)
        # source_batch_dataloader = DataLoader(source_data, sampler=source_sampler, batch_size=args.train_batch_size)



        '''load target k-shot data'''
        target_samples_features = convert_examples_to_features(
            target_samples_entail+target_samples_neutral + target_samples_contra,
            label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        target_samples_input_ids = torch.tensor([f.input_ids for f in target_samples_features], dtype=torch.long)
        target_samples_input_mask = torch.tensor([f.input_mask for f in target_samples_features], dtype=torch.long)
        target_samples_segment_ids = torch.tensor([f.segment_ids for f in target_samples_features], dtype=torch.long)
        target_samples_label_ids = torch.tensor([f.label_id for f in target_samples_features], dtype=torch.long)

        target_samples_data = TensorDataset(target_samples_input_ids, target_samples_input_mask, target_samples_segment_ids, target_samples_label_ids)
        target_samples_sampler = RandomSampler(target_samples_data)
        target_samples_dataloader = DataLoader(target_samples_data, sampler=target_samples_sampler, batch_size=9)
        target_samples_dataloader_batch_16 = DataLoader(target_samples_data, sampler=target_samples_sampler, batch_size=16)

        '''load dev set'''
        dev_examples = processor.get_RTE_as_dev('/export/home/Dataset/glue_data/RTE/dev.tsv')
        dev_features = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

        '''load test set'''
        eval_examples = processor.get_RTE_as_test('/export/home/Dataset/RTE/test_RTE_1235.txt')
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)



        logger.info("***** Running training *****")
        iter_co = 0
        tr_loss = 0
        loss_fct = CrossEntropyLoss()

        source_size = source_all_input_ids.shape[0]
        source_id_list = list(range(source_size))
        source_batch_size = 10
        source_batch_start = [x*source_batch_size for x in range(source_size//source_batch_size)]

        target_sample_size = target_samples_input_ids.shape[0]
        target_sample_id_list = list(range(target_sample_size))
        target_sample_batch_size = 9
        target_sample_batch_start = [x*target_sample_batch_size for x in range(target_sample_size//target_sample_batch_size)]


        max_test_acc = 0.0
        max_dev_acc = 0.0

        for _ in trange(int(args.NN_epochs), desc="NN Epoch"):
            '''for each epoch, we do 100 iter of NN; then full iter of target classification'''
            '''NN training Phase'''
            random.shuffle(source_id_list)
            random.shuffle(target_sample_id_list)

            source_sample_entail_reps_history = []
            source_sample_neutral_reps_history = []
            source_sample_contra_reps_history = []
            source_sample_entail_logits_history = []
            source_sample_neutral_logits_history = []
            source_sample_contra_logits_history = []
            for step, source_samples_batch in enumerate(source_samples_dataloader):

                source_samples_batch = tuple(t.to(device) for t in source_samples_batch)
                source_samples_input_ids, source_samples_input_mask, source_samples_segment_ids, source_samples_label_ids = source_samples_batch
                # assert input_ids.shape[0] == args.train_batch_size
                entail_size_i = (source_samples_label_ids==0)#.sum()
                neutral_size_i = (source_samples_label_ids==1)#.sum()
                contra_size_i = (source_samples_label_ids==2)#.sum()

                with torch.no_grad():
                    source_sample_logits, source_sample_reps = roberta_seq_model(source_samples_input_ids, source_samples_input_mask, None, labels=None)
                    source_sample_logits = source_sample_logits[0]
                    source_sample_reps = source_sample_reps#[:,0,:]
                source_sample_reps_logits = (source_sample_reps, source_sample_logits)


                source_sample_entail_reps_i = source_sample_reps[entail_size_i].mean(dim=0, keepdim=True)
                # print('source_sample_entail_reps_i:', source_sample_entail_reps_i)
                # print('entail_size_i:', entail_size_i, 'source_samples_label_ids:', source_samples_label_ids)
                # exit(0)
                source_sample_neutral_reps_i = source_sample_reps[neutral_size_i].mean(dim=0, keepdim=True)
                source_sample_contra_reps_i = source_sample_reps[contra_size_i].mean(dim=0, keepdim=True)

                source_sample_entail_logits_i = source_sample_logits[entail_size_i].mean(dim=0, keepdim=True)
                source_sample_neutral_logits_i = source_sample_logits[neutral_size_i].mean(dim=0, keepdim=True)
                source_sample_contra_logits_i = source_sample_logits[contra_size_i].mean(dim=0, keepdim=True)

                if entail_size_i.sum()!=0:
                    source_sample_entail_reps_history.append(source_sample_entail_reps_i)
                    source_sample_entail_logits_history.append(source_sample_entail_logits_i)

                if neutral_size_i.sum()!=0:
                    source_sample_neutral_reps_history.append(source_sample_neutral_reps_i)
                    source_sample_neutral_logits_history.append(source_sample_neutral_logits_i)

                if contra_size_i.sum()!=0:
                    source_sample_contra_reps_history.append(source_sample_contra_reps_i)
                    source_sample_contra_logits_history.append(source_sample_contra_logits_i)

                '''choose one batch in target samples'''
                selected_target_sample_start_list = random.sample(target_sample_batch_start, 1)
                # for start_i in selected_source_batch_start_list:
                start_i = selected_target_sample_start_list[0]
                ids_single = target_sample_id_list[start_i:start_i+target_sample_batch_size]
                #target_samples_input_ids, target_samples_input_mask, target_samples_segment_ids, target_samples_label_ids
                single_target_sample_input_ids = target_samples_input_ids[ids_single].to(device)
                single_target_sample_input_mask = target_samples_input_mask[ids_single].to(device)
                single_target_sample_segment_ids = target_samples_segment_ids[ids_single].to(device)
                single_target_sample_label_ids = target_samples_label_ids[ids_single].to(device)
                # single_input = (single_source_batch_input_ids, single_source_batch_input_mask, single_source_batch_segment_ids, single_source_batch_label_ids)
                with torch.no_grad():
                    target_sample_logits, target_sample_reps = roberta_seq_model(single_target_sample_input_ids, single_target_sample_input_mask, None, labels=None)
                target_sample_reps_logits_labels = (target_sample_reps, target_sample_logits[0], single_target_sample_label_ids)


                '''randomly select M batches from source'''
                selected_source_batch_start_list = random.sample(source_batch_start, 10)
                for start_i in selected_source_batch_start_list:
                    ids_single = source_id_list[start_i:start_i+source_batch_size]
                    #source_all_input_ids, source_all_input_mask, source_all_segment_ids, source_all_label_ids
                    single_source_batch_input_ids = source_all_input_ids[ids_single].to(device)
                    single_source_batch_input_mask = source_all_input_mask[ids_single].to(device)
                    single_source_batch_segment_ids = source_all_segment_ids[ids_single].to(device)
                    single_source_batch_label_ids = source_all_label_ids[ids_single].to(device)
                    # single_input = (single_source_batch_input_ids, single_source_batch_input_mask, single_source_batch_segment_ids, single_source_batch_label_ids)
                    with torch.no_grad():
                        _, source_batch_reps = roberta_seq_model(single_source_batch_input_ids, single_source_batch_input_mask, None, labels=None)
                    source_batch_reps_labels = (source_batch_reps, single_source_batch_label_ids)

                    model.train()
                    loss_nn = model(target_sample_reps_logits_labels, None, source_sample_reps_logits, source_batch_reps_labels,
                                                None, None, None, None, mode='train_NN', loss_fct = loss_fct)
                    # print('loss_nn:  ', loss_nn.item())
                    loss_nn.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if step == args.NN_iter_limit:#100:
                    break

            # print('source_sample_entail_reps_history before:', source_sample_entail_reps_history)

            source_sample_entail_reps_history = torch.cat(source_sample_entail_reps_history, dim=0).mean(dim=0, keepdim=True)
            source_sample_neutral_reps_history = torch.cat(source_sample_neutral_reps_history, dim=0).mean(dim=0, keepdim=True)
            source_sample_contra_reps_history = torch.cat(source_sample_contra_reps_history, dim=0).mean(dim=0, keepdim=True)
            source_sample_entail_logits_history = torch.cat(source_sample_entail_logits_history, dim=0).mean(dim=0, keepdim=True)
            source_sample_neutral_logits_history = torch.cat(source_sample_neutral_logits_history, dim=0).mean(dim=0, keepdim=True)
            source_sample_contra_logits_history = torch.cat(source_sample_contra_logits_history, dim=0).mean(dim=0, keepdim=True)

            # print('source_sample_entail_reps_history:', source_sample_entail_reps_history.shape)
            source_sample_reps_history = torch.cat([source_sample_entail_reps_history, source_sample_neutral_reps_history, source_sample_contra_reps_history], dim=0)
            source_sample_logits_history = torch.cat([source_sample_entail_logits_history, source_sample_neutral_logits_history, source_sample_contra_logits_history], dim=0)
            source_reps_logits_history = (source_sample_reps_history, source_sample_logits_history)
            # print('source_sample_logits_history:', source_sample_logits_history)

            '''STILTS Phase, train target classifier'''
            iter_co = 0
            for stilts_epoch in trange(int(args.stilts_epochs), desc="STILTS Epoch"):
                target_sample_entail_reps_history_list = []
                target_sample_neutral_reps_history_list = []
                target_sample_contra_reps_history_list = []
                target_sample_entail_logits_history_list = []
                target_sample_neutral_logits_history_list = []
                target_sample_contra_logits_history_list = []

                # for target_sample_batch in target_samples_dataloader:
                for target_sample_batch in target_samples_dataloader_batch_16:
                    target_sample_batch = tuple(t.to(device) for t in target_sample_batch)
                    target_sample_input_ids_batch, target_sample_input_mask_batch, target_sample_segment_ids_batch, target_sample_label_ids_batch = target_sample_batch
                    # assert input_ids.shape[0] == args.train_batch_size
                    entail_size_i = (target_sample_label_ids_batch==0)#.sum()
                    neutral_size_i = (target_sample_label_ids_batch==1)#.sum()
                    contra_size_i = (target_sample_label_ids_batch==2)#.sum()
                    with torch.no_grad():
                        # print('roberta_seq_model config:', roberta_seq_model.config)
                        target_sample_logits_tuple, target_sample_reps = roberta_seq_model(target_sample_input_ids_batch, target_sample_input_mask_batch, None, labels=None)
                        # print('target_sample_logits_tuple:', target_sample_logits_tuple)
                        # exit(0)
                        target_sample_logits = target_sample_logits_tuple[0]
                        # print('len(target_sample_logits_tuple):', len(target_sample_logits_tuple))
                        assert len(target_sample_logits_tuple) == 2 #(logits, (hidden_states)
                        # target_sample_last3_reps = torch.cat([target_sample_logits_tuple[1][-4][:,0,:], target_sample_logits_tuple[1][-3][:,0,:], target_sample_logits_tuple[1][-2][:,0,:]], dim=1) #(batch, 1024*3)
                        # target_sample_last3_reps = torch.cat([torch.mean(target_sample_logits_tuple[1][-4], dim=1), torch.mean(target_sample_logits_tuple[1][-3], dim=1), torch.mean(target_sample_logits_tuple[1][-2], dim=1)], dim=1) #(batch, 1024*3)
                        target_sample_last3_reps = torch.mean(target_sample_logits_tuple[1][-7], dim=1) +torch.mean(target_sample_logits_tuple[1][-6], dim=1) +torch.mean(target_sample_logits_tuple[1][-5], dim=1) +torch.mean(target_sample_logits_tuple[1][-4], dim=1) + torch.mean(target_sample_logits_tuple[1][-3], dim=1) + torch.mean(target_sample_logits_tuple[1][-2], dim=1)
                        target_sample_reps = target_sample_reps#[:,0,:]
                        # target_sample_reps = torch.mean(target_sample_reps, dim=1)
                    target_sample_reps_logits_labels = (target_sample_reps, target_sample_logits, target_sample_label_ids_batch)


                    target_sample_entail_reps_i = target_sample_reps[entail_size_i].mean(dim=0, keepdim=True)
                    target_sample_neutral_reps_i = target_sample_reps[neutral_size_i].mean(dim=0, keepdim=True)
                    target_sample_contra_reps_i = target_sample_reps[contra_size_i].mean(dim=0, keepdim=True)

                    target_sample_entail_logits_i = target_sample_logits[entail_size_i].mean(dim=0, keepdim=True)
                    target_sample_neutral_logits_i = target_sample_logits[neutral_size_i].mean(dim=0, keepdim=True)
                    target_sample_contra_logits_i = target_sample_logits[contra_size_i].mean(dim=0, keepdim=True)

                    if entail_size_i.sum()!=0:
                        target_sample_entail_reps_history_list.append(target_sample_entail_reps_i)
                        target_sample_entail_logits_history_list.append(target_sample_entail_logits_i)
                    if neutral_size_i.sum()!=0:
                        target_sample_neutral_reps_history_list.append(target_sample_neutral_reps_i)
                        target_sample_neutral_logits_history_list.append(target_sample_neutral_logits_i)
                    if contra_size_i.sum()!=0:
                        target_sample_contra_reps_history_list.append(target_sample_contra_reps_i)
                        target_sample_contra_logits_history_list.append(target_sample_contra_logits_i)


                    model.train()
                    loss_cl = model(target_sample_reps_logits_labels, target_sample_last3_reps, None, None,
                                                None, None, None, None, mode='train_CL', loss_fct = loss_fct)
                    # print('loss_cl:  ', loss_cl.item())
                    loss_cl.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    iter_co+=1
                    if iter_co % 100 ==0:
                        '''dev or test'''
                        if (len(target_sample_entail_reps_history_list)==0 or
                            len(target_sample_neutral_reps_history_list)==0 or
                            len(target_sample_contra_reps_history_list)==0):
                            '''train next target_sample batch'''
                            continue

                        target_sample_entail_reps_history = torch.cat(target_sample_entail_reps_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_neutral_reps_history = torch.cat(target_sample_neutral_reps_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_contra_reps_history = torch.cat(target_sample_contra_reps_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_entail_logits_history = torch.cat(target_sample_entail_logits_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_neutral_logits_history = torch.cat(target_sample_neutral_logits_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_contra_logits_history = torch.cat(target_sample_contra_logits_history_list, dim=0).mean(dim=0, keepdim=True)
                        target_sample_reps_history = torch.cat([target_sample_entail_reps_history, target_sample_neutral_reps_history, target_sample_contra_reps_history], dim=0)
                        target_sample_logits_history = torch.cat([target_sample_entail_logits_history, target_sample_neutral_logits_history, target_sample_contra_logits_history], dim=0)
                        target_reps_logits_history = (target_sample_reps_history, target_sample_logits_history)
                        # print('target_sample_logits_history:', target_sample_logits_history)

                        '''
                        start evaluate on dev set after this epoch
                        '''
                        model.eval()
                        for idd, dev_or_test_dataloader in enumerate([dev_dataloader, eval_dataloader]):

                            preds = []
                            preds_NN = []
                            preds_pre = []
                            preds_CL = []
                            gold_label_ids = []
                            print('Evaluating...')
                            for input_ids, input_mask, segment_ids, label_ids in dev_or_test_dataloader:
                                input_ids = input_ids.to(device)
                                input_mask = input_mask.to(device)
                                segment_ids = segment_ids.to(device)
                                label_ids = label_ids.to(device)
                                # gold_label_ids+=list(label_ids.detach().cpu().numpy())

                                with torch.no_grad():
                                    test_batch_logits_tuple, test_batch_reps = roberta_seq_model(input_ids, input_mask, None, labels=None)
                                    test_batch_logits = test_batch_logits_tuple[0]
                                    test_batch_reps = test_batch_reps#[:,0,:]
                                    # test_batch_reps = torch.mean(test_batch_reps, dim=1)
                                    assert len(test_batch_logits_tuple) == 2 #(logits, (hidden_states), (attentions))
                                    # test_batch_last3_reps = torch.cat([test_batch_logits_tuple[1][-4][:,0,:], test_batch_logits_tuple[1][-3][:,0,:], test_batch_logits_tuple[1][-2][:,0,:]], dim=1) #(batch, 1024*3)
                                    # test_batch_last3_reps = torch.cat([torch.mean(test_batch_logits_tuple[1][-4], dim=1), torch.mean(test_batch_logits_tuple[1][-3], dim=1), torch.mean(test_batch_logits_tuple[1][-2], dim=1)], dim=1) #(batch, 1024*3)
                                    test_batch_last3_reps = torch.mean(test_batch_logits_tuple[1][-7], dim=1) +torch.mean(test_batch_logits_tuple[1][-6], dim=1) +torch.mean(test_batch_logits_tuple[1][-5], dim=1) +torch.mean(test_batch_logits_tuple[1][-4], dim=1) + torch.mean(test_batch_logits_tuple[1][-3], dim=1) + torch.mean(test_batch_logits_tuple[1][-2], dim=1)
                                    test_batch_reps_logits_labels = (test_batch_reps,test_batch_logits, label_ids)

                                    pred_labels_i,  pred_labels_NN_i, pred_labels_pre_i, pred_labels_CL_i= model(None, None, None, None,
                                                        test_batch_reps_logits_labels, test_batch_last3_reps, source_reps_logits_history, target_reps_logits_history,
                                                        mode='test', loss_fct = loss_fct)
                                # print('pred_labels_i:',pred_labels_i)
                                preds.append(pred_labels_i)

                                preds_NN.append(pred_labels_NN_i)
                                preds_pre.append(pred_labels_pre_i)
                                preds_CL.append(pred_labels_CL_i)

                                gold_label_ids.append(label_ids)
                            # print('preds:', preds)

                            test_acc = acc_calculate(preds, gold_label_ids)
                            acc_nn = acc_calculate(preds_NN, gold_label_ids)
                            acc_pre = acc_calculate(preds_pre, gold_label_ids)
                            acc_cl = acc_calculate(preds_CL, gold_label_ids)
                            fine_grain_acc_list= [acc_nn, acc_pre, acc_cl]
                            # pred_label_ids = torch.cat(preds,dim=0).detach().cpu().numpy()
                            # gold_label_ids = torch.cat(gold_label_ids,dim=0).detach().cpu().numpy()




                            # pred_label_ids_binary = []
                            # for p in pred_label_ids:
                            #     pred_label_ids_binary.append(0 if p == 0 else 1)
                            # gold_label_ids_binary = []
                            # for p in gold_label_ids:
                            #     gold_label_ids_binary.append(0 if p == 0 else 1)
                            #
                            # assert len(pred_label_ids_binary) ==  len(gold_label_ids_binary)
                            #
                            # print('pred neg ratio:', sum(pred_label_ids_binary)/len(pred_label_ids_binary))
                            # print('gold neg ratio:', sum(gold_label_ids_binary)/len(gold_label_ids_binary))
                            # hit_co = 0
                            # for k in range(len(pred_label_ids_binary)):
                            #     if pred_label_ids_binary[k] == gold_label_ids_binary[k]:
                            #         hit_co +=1
                            # test_acc = hit_co/len(gold_label_ids_binary)


                            if idd == 0: # this is dev
                                if test_acc > max_dev_acc:
                                    max_dev_acc = test_acc
                                    print(stilts_epoch, ' dev acc:', test_acc, fine_grain_acc_list, ' max_mean_dev_acc:', max_dev_acc, '\n')
                                    '''store the model'''
                                    # store_transformers_models(model, tokenizer, '/export/home/Dataset/BERT_pretrained_mine/crossdataentail/trainMNLItestRTE', str(max_dev_acc))

                                else:
                                    print(stilts_epoch, ' dev acc:', test_acc, fine_grain_acc_list, ' max_mean_dev_acc:', max_dev_acc, '\n')
                                    # break
                            else: # this is test
                                if test_acc > max_test_acc:
                                    max_test_acc = test_acc
                                print(stilts_epoch, ' test acc:', test_acc, fine_grain_acc_list, ' max_test_acc:', max_test_acc, '\n')

def acc_calculate(pred_label_ids_I, gold_label_ids_I):
    pred_label_ids_I = torch.cat(pred_label_ids_I,dim=0).detach().cpu().numpy()
    gold_label_ids_I = torch.cat(gold_label_ids_I,dim=0).detach().cpu().numpy()

    pred_label_ids_binary = []
    for p in pred_label_ids_I:
        pred_label_ids_binary.append(0 if p == 0 else 1)
    gold_label_ids_binary = []
    for p in gold_label_ids_I:
        gold_label_ids_binary.append(0 if p == 0 else 1)

    assert len(pred_label_ids_binary) ==  len(gold_label_ids_binary)

    print('pred neg ratio:', sum(pred_label_ids_binary)/len(pred_label_ids_binary))
    print('gold neg ratio:', sum(gold_label_ids_binary)/len(gold_label_ids_binary))
    hit_co = 0
    for k in range(len(pred_label_ids_binary)):
        if pred_label_ids_binary[k] == gold_label_ids_binary[k]:
            hit_co +=1
    test_acc = hit_co/len(gold_label_ids_binary)
    return test_acc


if __name__ == "__main__":
    main()
    # a = torch.randn(4, 4)
    # print(torch.argmax(a, dim=1, keepdim=True))


    '''
    我们改了关于shuffle, 和sampling_seed的错误；
    发现 MNLI只load 2000的时候NN 是最好的，基本上81.99%， 但是偶数epoch dev比test高三个点，但是奇数epoch dev则更低
    load全部的话刚开始只有80%
    '''
# CUDA_VISIBLE_DEVICES=3 python -u 2019to2020_train_MNLI_kshot_RTE.py --task_name rte --do_train --do_lower_case --bert_model bert-large-uncased --learning_rate 1e-5 --data_dir '' --output_dir '' --k_shot 3 --sampling_seed 42 --NN_epochs 1 --NN_iter_limit 100 --stilts_epochs 20
