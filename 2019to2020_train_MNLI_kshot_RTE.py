
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

        '''now randomly sampling'''
        entail_size = len(entail_list)
        not_entail_size = len(not_entail_list)
        print('entail_size:', entail_size, 'not_entail_size:', not_entail_size)
        if K <= entail_size:
            sampled_entail = random.Random(sampling_seed).sample(entail_list, K)
        else:
            sampled_entail = random.Random(sampling_seed).choices(entail_list, k = K)
        if K <= int(not_entail_size/2):
            sampled_not_entail = random.Random(sampling_seed).sample(not_entail_list, 2*K)
        else:
            sampled_not_entail = random.Random(sampling_seed).choices(not_entail_list, k = 2*K)

        # print('sampled_entail size:', len(sampled_entail))
        # print('sampled_not_entail size:', len(sampled_not_entail))
        examples_entail=[]
        examples_neutral=[]
        examples_contra=[]

        for idd, pair in enumerate(sampled_entail):
            examples_entail.append(
                InputExample(guid='entail_'+str(idd), text_a=pair[0], text_b=pair[1], label='entailment'))

        for idd, pair in enumerate(sampled_not_entail):
            if idd < K:
                '''neutral'''
                examples_neutral.append(
                    InputExample(guid='neutral_'+str(idd), text_a=pair[0], text_b=pair[1], label='neutral'))
            else:
                '''contradiction'''
                examples_contra.append(
                    InputExample(guid='contra_'+str(idd), text_a=pair[0], text_b=pair[1], label='contradiction'))

        assert len(examples_entail) == K
        assert len(examples_neutral) == K
        assert len(examples_contra) == K
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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

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
        '''??? why a different name will not get initialized'''
        # self.roberta = RobertaModel(config)
        '''classifier for target domain'''
        self.classifier = RobertaClassificationHead(config)

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

        print('sample_size:', sample_size, 'query_size:', query_size, 'hidden_size:', hidden_size)


        repeat_sample_rep = torch.cat([sample_reps]*query_size, dim=0) #(9*batch_size, hidden)
        print('repeat_sample_rep shape:', repeat_sample_rep.shape)
        print('query_reps:', query_reps.shape)
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



    def forward(self, target_sample_reps_logits_labels, source_sample_reps_logits, source_batch_reps_labels,
                test_batch_reps_logits, source_reps_logits_history, target_reps_logits_history,
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
            test_batch_reps, test_batch_logits = test_batch_reps_logits
            source_sample_reps_history, source_sample_logits_history = source_reps_logits_history
            target_sample_reps_history, target_sample_logits_history = target_reps_logits_history


        if mode == 'train_NN':
            loss_target_pred_source = self.NearestNeighbor(target_sample_reps, target_sample_logits, source_batch_reps, source_batch_labels, mode='train_NN', loss_fct = loss_fct)
            loss_source_pred_source = self.NearestNeighbor(source_sample_reps, source_sample_logits, source_batch_reps, source_batch_labels, mode='train_NN', loss_fct = loss_fct)
            loss_source_pred_target = self.NearestNeighbor(source_sample_reps, source_sample_logits, target_sample_reps, target_sample_labels, mode='train_NN', loss_fct = loss_fct)

            NN_loss = loss_target_pred_source+loss_source_pred_source+loss_source_pred_target
            return NN_loss
        elif mode == 'train_CL':
            target_sample_CL_logits = self.classifier(target_sample_reps)
            CL_loss = loss_fct(target_sample_CL_logits.view(-1, self.num_labels), target_sample_labels.view(-1))
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
            CL_logits_from_target = self.classifier(test_batch_reps)

            overall_test_batch_logits = logits_from_pretrained+NN_logits_combine+CL_logits_from_target
            return overall_test_batch_logits




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
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
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
    parser.add_argument("--num_train_epochs",
                        default=3.0,
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
    roberta_seq_model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    roberta_seq_model.to(device)
    roberta_seq_model.eval()

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

        iter_co = 0

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            '''for each epoch, we do 100 iter of NN; then full iter of target classification'''
            '''NN training'''
            random.Random(args.sampling_seed).shuffle(source_id_list)
            for step, source_samples_batch in enumerate(source_samples_dataloader):

                source_samples_batch = tuple(t.to(device) for t in source_samples_batch)
                source_samples_input_ids, source_samples_input_mask, source_samples_segment_ids, source_samples_label_ids = source_samples_batch
                # assert input_ids.shape[0] == args.train_batch_size
                with torch.no_grad():
                    source_sample_logits, source_sample_reps = roberta_seq_model(source_samples_input_ids, source_samples_input_mask, None, labels=None)
                source_sample_reps_logits = (source_sample_reps, source_sample_logits)

                '''choose one batch in target samples'''
                selected_target_sample_start_list = random.Random(args.sampling_seed).sample(target_sample_batch_start, 1)
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
                target_sample_reps_logits_labels = (target_sample_reps, target_sample_logits, single_target_sample_label_ids)


                '''randomly select M batches from source'''


                selected_source_batch_start_list = random.Random(args.sampling_seed).sample(source_batch_start, 5)
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
                    loss_nn = model(target_sample_reps_logits_labels, source_sample_reps_logits, source_batch_reps_labels,
                                                None, None, None, mode='train_NN', loss_fct = loss_fct)
                    print('loss_nn:  ', loss_nn.item())
                    loss_nn.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if step == 5:#100:
                    break
            '''now, train target classifier'''
            for target_sample_batch in target_samples_dataloader:
                target_sample_batch = tuple(t.to(device) for t in target_sample_batch)
                target_sample_input_ids_batch, target_sample_input_mask_batch, target_sample_segment_ids_batch, target_sample_label_ids_batch = target_sample_batch
                # assert input_ids.shape[0] == args.train_batch_size
                with torch.no_grad():
                    target_sample_logits, target_sample_reps = roberta_seq_model(target_sample_input_ids_batch, target_sample_input_mask_batch, None, labels=None)
                target_sample_reps_logits_labels = (target_sample_reps, target_sample_logits, target_sample_label_ids_batch)

                model.train()
                loss_cl = model(target_sample_reps_logits_labels, None, None,
                                            None, None, None, mode='train_CL', loss_fct = loss_fct)
                print('loss_cl:  ', loss_cl.item())
                loss_cl.backward()
                optimizer.step()
                optimizer.zero_grad()


                iter_co+=1
                # if iter_co % 50 ==0:
                #     '''dev or test'''


                    # def forward(self, target_sample_reps_logits_labels, source_sample_reps_logits, source_batch_reps_labels,
                    #             test_batch_reps_logits, source_reps_logits_history, target_reps_logits_history,
                    #             mode='train_NN'):


                #
                # iter_co+=1
                #
                # check_freq = 10
                # if iter_co %check_freq==0:
                #     '''first get info from MNLI by sampling'''
                #     assert len(sample_input_ids_each_iter) == check_freq
                #     mnli_sample_hidden_list = []
                #     mnli_sample_logits_list = []
                #     for ff in range(len(sample_input_ids_each_iter)):
                #         model.eval()
                #         with torch.no_grad():
                #             mnli_sample_hidden_i, mnli_sample_logits_i = model(sample_input_ids_each_iter[ff], None, sample_input_mask_each_iter[ff], sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs = None, few_shot_training=False, is_train=False, fetch_hidden_only=True, loss_fct=None)
                #             mnli_sample_hidden_list.append(mnli_sample_hidden_i[None,:,:])
                #             mnli_sample_logits_list.append(mnli_sample_logits_i[None,:,:])
                #     sample_input_ids_each_iter = []
                #     sample_input_mask_each_iter = []
                #     '''sum or mean does not make big difference'''
                #     prior_mnli_samples_outputs = torch.cat(mnli_sample_hidden_list,dim=0)
                #     prior_mnli_samples_outputs = torch.mean(prior_mnli_samples_outputs,dim=0)
                #     prior_mnli_samples_logits = torch.cat(mnli_sample_logits_list,dim=0)
                #     prior_mnli_samples_logits = torch.mean(prior_mnli_samples_logits,dim=0)
                #
                #     '''second do few-shot training'''
                #     for ff in range(2):
                #         model.train()
                #         few_loss = model(eval_all_input_ids_shot.to(device), None, eval_all_input_mask_shot.to(device), sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,1,1,1]), prior_samples_outputs = None, few_shot_training=True, is_train=True, loss_fct=loss_fct)
                #         few_loss.backward()
                #         optimizer.step()
                #         optimizer.zero_grad()
                #         print('few_loss:', few_loss)
                #     '''
                #     start evaluate on dev set after this epoch
                #     '''
                #     model.eval()
                #     for idd, dev_or_test_dataloader in enumerate([dev_dataloader, eval_dataloader]):
                #         logger.info("***** Running evaluation *****")
                #         if idd == 0:
                #             logger.info("  Num examples = %d", len(dev_examples))
                #         else:
                #             logger.info("  Num examples = %d", len(eval_examples))
                #         logger.info("  Batch size = %d", args.eval_batch_size)
                #
                #         eval_loss = 0
                #         nb_eval_steps = 0
                #         preds = []
                #         preds_LR= []
                #         preds_NN = []
                #         gold_label_ids = []
                #         print('Evaluating...')
                #         for input_ids, input_mask, segment_ids, label_ids in dev_or_test_dataloader:
                #             input_ids = input_ids.to(device)
                #             input_mask = input_mask.to(device)
                #             segment_ids = segment_ids.to(device)
                #             label_ids = label_ids.to(device)
                #             gold_label_ids+=list(label_ids.detach().cpu().numpy())
                #
                #             all_input_ids = torch.cat([eval_all_input_ids_shot.to(device),input_ids],dim=0)
                #             all_input_mask = torch.cat([eval_all_input_mask_shot.to(device),input_mask], dim=0)
                #
                #
                #             with torch.no_grad():
                #                 logits_LR, logits_NN, logits = model(all_input_ids, None, all_input_mask, sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs = prior_mnli_samples_outputs, prior_samples_logits = prior_mnli_samples_logits, is_train=False, loss_fct=None)
                #
                #             nb_eval_steps += 1
                #             if len(preds) == 0:
                #                 preds.append(logits.detach().cpu().numpy())
                #                 preds_LR.append(logits_LR.detach().cpu().numpy())
                #                 preds_NN.append(logits_NN.detach().cpu().numpy())
                #             else:
                #                 preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                #                 preds_LR[0] = np.append(preds_LR[0], logits_LR.detach().cpu().numpy(), axis=0)
                #                 preds_NN[0] = np.append(preds_NN[0], logits_NN.detach().cpu().numpy(), axis=0)
                #
                #         preds = preds[0]
                #         preds_LR = preds_LR[0]
                #         preds_NN = preds_NN[0]
                #
                #         '''
                #         preds: size*3 ["entailment", "neutral", "contradiction"]
                #         wenpeng added a softxmax so that each row is a prob vec
                #         '''
                #         acc_list = []
                #         for preds_i in [preds_LR, preds_NN, preds]:
                #             pred_probs = softmax(preds_i,axis=1)
                #             pred_indices = np.argmax(pred_probs, axis=1)
                #             pred_label_ids = []
                #             for p in pred_indices:
                #                 pred_label_ids.append(0 if p == 0 else 1)
                #             gold_label_ids = gold_label_ids
                #             assert len(pred_label_ids) == len(gold_label_ids)
                #             hit_co = 0
                #             for k in range(len(pred_label_ids)):
                #                 if pred_label_ids[k] == gold_label_ids[k]:
                #                     hit_co +=1
                #             test_acc = hit_co/len(gold_label_ids)
                #
                #             acc_list.append(test_acc)
                #
                #
                #         softmax_LR = array_2_softmax(preds_LR)
                #         softmax_NN = array_2_softmax(preds_NN)
                #         preds_ensemble = []
                #         for i in range(softmax_LR.shape[0]):
                #             if softmax_LR[i][0] > softmax_LR[i][1] and softmax_NN[i][0] > softmax_NN[i][1]:
                #                 preds_ensemble.append(0)
                #             elif softmax_LR[i][0] < softmax_LR[i][1] and softmax_NN[i][0] < softmax_NN[i][1]:
                #                 preds_ensemble.append(1)
                #             elif softmax_LR[i][0] > softmax_LR[i][1] and softmax_LR[i][0] > softmax_NN[i][1]:
                #                 preds_ensemble.append(0)
                #             else:
                #                 preds_ensemble.append(1)
                #         hit_co = 0
                #         for k in range(len(preds_ensemble)):
                #             if preds_ensemble[k] == gold_label_ids[k]:
                #                 hit_co +=1
                #         test_acc = hit_co/len(gold_label_ids)
                #         acc_list.append(test_acc)
                #
                #
                #         if idd == 0: # this is dev
                #             if acc_list[0] >= max_dev_acc:
                #                 max_dev_acc = acc_list[0]
                #                 print('\ndev acc_list:', acc_list, ' max_mean_dev_acc:', max_dev_acc, '\n')
                #                 '''store the model'''
                #                 # store_transformers_models(model, tokenizer, '/export/home/Dataset/BERT_pretrained_mine/crossdataentail/trainMNLItestRTE', str(max_dev_acc))
                #
                #             else:
                #                 print('\ndev acc_list:', acc_list, ' max_dev_acc:', max_dev_acc, '\n')
                #                 break
                #         else: # this is test
                #             if acc_list[-2] > max_test_acc:
                #                 max_test_acc = acc_list[-2]
                #             print('\ntest acc_list:', acc_list, ' max_test_acc:', max_test_acc, '\n')
                #

def array_2_softmax(a):
    for i in range(a.shape[0]):
        if a[i][2]>a[i][1]:
            a[i][1] = a[i][2]
    sub_a = a[:,:2]
    return softmax(sub_a)

if __name__ == "__main__":
    main()
    '''
    这儿针对以前的train_MNLI_test_3shotRTE_meta_learning.v2.py的改动就是在RTE sampling的时候；
    然后就是只care acc_list[0]的测试标准，
    '''
# CUDA_VISIBLE_DEVICES=3 python -u 2019to2020_train_MNLI_kshot_3shot_RTE.py --task_name rte --do_train --do_lower_case --bert_model bert-large-uncased --learning_rate 1e-5 --data_dir '' --output_dir '' --k_shot 3 --sampling_seed 42
