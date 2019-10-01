
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



from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.modeling_roberta import RobertaModel, RobertaConfig#, RobertaClassificationHead
from pytorch_transformers.modeling_bert import BertPreTrainedModel

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
            # if line_co > 20000:
            #     break
        readfile.close()
        print('loaded  size:', line_co)
        return examples_entail, examples_neutral, examples_contra




    def get_SciTail_as_train(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples_entail=[]
        examples_neutral=[]
        examples_contra=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        class2size = defaultdict(int)
        line_co=0
        for row in readfile:
            line=row.strip().split('\t')
            if len(line) == 3:
                guid = "3shot-"+str(line_co)
                text_a = line[0].strip()
                text_b = line[1].strip()
                random_value = random.uniform(0, 1)
                if  random_value < 0.45:
                    continue
                # label = line[3].strip() #["entailment", "not_entailment"]
                # label = 'entailment'  if line[3].strip() == 'entailment' else 'neutral'
                if line[2].strip() == 'entails':
                    labels = ['entailment']
                else:
                    labels = ['neutral', 'contradiction']
                for label in labels:
                    if class2size.get(label, 0) < 3:
                        if label == 'entailment':
                            examples_entail.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        elif label == 'neutral':
                            examples_neutral.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        else:
                            examples_contra.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                        class2size[label]+=1
                    else:
                        continue
                if len(class2size.keys()) == 3 and sum(class2size.values()) == 9:
                    break
                line_co+=1
        readfile.close()
        print('loaded  3shot size:', line_co)
        assert len(examples_entail) == 3
        assert len(examples_neutral) == 3
        assert len(examples_contra) == 3
        return examples_entail, examples_neutral, examples_contra

    def get_SciTail_as_dev_or_test(self, filename, prefix):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:

            line=row.strip().split('\t')
            if len(line) == 3:
                guid = prefix+'-'+str(line_co-1)
                text_a = line[0].strip()
                text_b = line[1].strip()
                # label = line[3].strip() #["entailment", "not_entailment"]
                label = 'entailment'  if line[2] == 'entails' else 'neutral'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
        readfile.close()
        print('loaded  size:', line_co-1)
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
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp_1 = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.mlp_2 = nn.Linear(config.hidden_size, 1, bias=False)
        # self.init_weights()
        # self.apply(self.init_bert_weights)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sample_size=None, class_size = None, labels=None, sample_labels=None, prior_samples_outputs=None, prior_samples_logits = None, few_shot_training=False, is_train = True, fetch_hidden_only=False, loss_fct = None):

        '''
        samples: input_ids, token_type_ids, attention_mask; in class order
        minibatch: input_ids, token_type_ids, attention_mask
        '''
        # print('input_ids shape0 :', input_ids.shape[0])
        outputs = self.roberta(input_ids, token_type_ids, attention_mask) #(batch, max_len, hidden_size)
        pooled_outputs = outputs[1]#torch.max(outputs[0],dim=1)[0]+ outputs[1]#outputs[1]#torch.mean(outputs[0],dim=1)#outputs[1] #(batch, hidden_size)
        LR_logits = self.classifier(pooled_outputs) #(9+batch, 3)
        if is_train:

            '''??? output samples_outputs for accumulating info for testing phase'''
            samples_outputs = pooled_outputs[:sample_size*class_size,:] #(9, hidden_size)
            '''make the dot prod between samples to zero'''
            samples_outputs_2_class_rep = samples_outputs.reshape(3,3,samples_outputs.shape[1])
            samples_outputs_2_class_rep = torch.sum(samples_outputs_2_class_rep,dim=1) #(3, hidden)
            class_dot_prod = nn.Sigmoid()(torch.mm(samples_outputs_2_class_rep, samples_outputs_2_class_rep.t()) )#(3,3)
            loss_cmu = torch.sum((class_dot_prod - torch.cuda.eye(3))**2)
            '''we use all into LR'''

            sample_logits = LR_logits[:sample_size*class_size,:] #(9,3)
            batch_logits_from_LR = nn.Softmax(dim=1)(LR_logits[sample_size*class_size:,:]) #(10,3)
            # if few_shot_training:


            '''This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.'''
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            sample_loss = loss_fct(sample_logits.view(-1, self.num_labels), sample_labels.view(-1))
            if few_shot_training:
                return sample_loss
            '''nearest neighber'''
            batch_outputs = pooled_outputs[sample_size*class_size:,:] #(batch, hidden_size)

            batch_size = batch_outputs.shape[0]
            hidden_size = batch_outputs.shape[1]
            # print('batch_size:',batch_size, 'hidden_size:', hidden_size)


            # samples_outputs = samples_outputs.reshape(sample_size, class_size, samples_outputs.shape[1])
            '''we use average for class embedding'''
            # class_rep = torch.mean(samples_outputs,dim=0) #(class_size, hidden_size)
            repeat_sample_rep = torch.cat([samples_outputs]*batch_size, dim=0) #(9*batch_size, hidden)




            # repeat_batch_outputs = tile(batch_outputs,0,class_size) #(batch*class_size, hidden)
            repeat_batch_outputs = batch_outputs.repeat(1, samples_outputs.shape[0]).view(-1, hidden_size)#(9*batch_size, hidden)
            '''? add similarity or something similar?'''
            mlp_input = torch.cat([
            repeat_batch_outputs, repeat_sample_rep,
            # repeat_batch_outputs - repeat_sample_rep,
            # cosine_rowwise_two_matrices(repeat_batch_outputs, repeat_sample_rep),
            repeat_batch_outputs*repeat_sample_rep
            ], dim=1) #(batch*class_size, hidden*2)
            '''??? add drop out here'''
            group_scores = torch.tanh(self.mlp_2(self.dropout(torch.tanh(self.mlp_1(self.dropout(mlp_input))))))#(batch*class_size, 1)
            group_scores_with_simi = group_scores + cosine_rowwise_two_matrices(repeat_batch_outputs, repeat_sample_rep)
            # group_scores = torch.tanh(self.mlp_2((torch.tanh(mlp_input))))#(9*batch_size, 1)
            # print('group_scores:',group_scores)

            similarity_matrix = group_scores_with_simi.reshape(batch_size, samples_outputs.shape[0])
            '''???note that the softmax will make the resulting logits smaller than LR'''
            batch_logits_from_NN = torch.mm(nn.Softmax(dim=1)(similarity_matrix), sample_logits) #(batch, 3)
            '''???use each of the logits for loss compute'''
            batch_logits = batch_logits_from_LR+batch_logits_from_NN


            '''??? add bias here'''

            '''This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.'''
            batch_loss = (loss_fct(batch_logits_from_LR.view(-1, self.num_labels), labels.view(-1))+
                        loss_fct(batch_logits_from_NN.view(-1, self.num_labels), labels.view(-1)))
            loss = sample_loss+batch_loss+loss_cmu
            return loss, samples_outputs

        else:
            '''testing'''
            batch_logits_from_LR = nn.Softmax(dim=1)(LR_logits[sample_size*class_size:,:]) #(10,3)


            '''??? output samples_outputs for accumulating info for testing phase'''
            samples_outputs = pooled_outputs[:sample_size*class_size,:] #(9, hidden_size)
            if fetch_hidden_only:
                return samples_outputs, LR_logits[:sample_size*class_size,:]


            samples_outputs =  torch.cat([prior_samples_outputs, samples_outputs], dim=0)
            batch_outputs = pooled_outputs[sample_size*class_size:,:] #(batch, hidden_size)
            # print('samples_outputs shaoe:', samples_outputs.shape)
            # sample_logits = self.classifier(samples_outputs) #(9, 3)
            # if few_shot_training:
            #     loss_fct = CrossEntropyLoss()
            #     '''This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.'''
            #     # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #     sample_loss = loss_fct(sample_logits.view(-1, self.num_labels), sample_labels.view(-1))
            #     return sample_loss


            batch_size = batch_outputs.shape[0]
            hidden_size = batch_outputs.shape[1]
            # print('batch_size:',batch_size, 'hidden_size:', hidden_size)


            # samples_outputs = samples_outputs.reshape(sample_size, class_size, samples_outputs.shape[1])
            '''we use average for class embedding'''
            # class_rep = torch.mean(samples_outputs,dim=0) #(class_size, hidden_size)
            repeat_sample_rep = torch.cat([samples_outputs]*batch_size, dim=0) #(9*batch_size, hidden)
            repeat_batch_outputs = batch_outputs.repeat(1, samples_outputs.shape[0]).view(-1, hidden_size)#(9*batch_size, hidden)
            '''? add similarity or something similar?'''
            mlp_input = torch.cat([
            repeat_batch_outputs, repeat_sample_rep,
            # repeat_batch_outputs - repeat_sample_rep,
            # cosine_rowwise_two_matrices(repeat_batch_outputs, repeat_sample_rep),
            repeat_batch_outputs*repeat_sample_rep
            ], dim=1) #(batch*class_size, hidden*2)
            '''??? add drop out here'''
            group_scores = torch.tanh(self.mlp_2(self.dropout(torch.tanh(self.mlp_1(self.dropout(mlp_input))))))#(batch*class_size, 1)
            group_scores_with_simi = group_scores + cosine_rowwise_two_matrices(repeat_batch_outputs, repeat_sample_rep)
            # group_scores = torch.tanh(self.mlp_2((torch.tanh(mlp_input))))#(9*batch_size, 1)
            # print('group_scores:',group_scores)

            similarity_matrix = group_scores_with_simi.reshape(batch_size, samples_outputs.shape[0])

            if prior_samples_logits is not None:
                sample_logits = torch.cuda.FloatTensor(9, 3).fill_(0)
                sample_logits[torch.arange(0, 9).long(), sample_labels] = 1.0
                sample_logits = sample_logits.repeat(2,1)
            else:
                '''the results now that using LR predicted logits is better'''
                sample_logits = prior_samples_logits
                sample_logits = torch.cat([sample_logits, LR_logits[:sample_size*class_size,:]],dim=0)
            batch_logits_from_NN = nn.Softmax(dim=1)(torch.mm(nn.Softmax(dim=1)(similarity_matrix), sample_logits)) #(batch, 3)
            # print('batch_logits_from_LR:',batch_logits_from_LR)
            # print('batch_logits_from_NN:', batch_logits_from_NN)
            logits = batch_logits_from_LR+batch_logits_from_NN
            return batch_logits_from_LR, batch_logits_from_NN, logits


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



    train_examples_entail, train_examples_neutral, train_examples_contra = processor.get_MNLI_as_train('/export/home/Dataset/glue_data/MNLI/train.tsv') #train_pu_half_v1.txt
    train_examples_entail_RTE, train_examples_neutral_RTE, train_examples_contra_RTE = processor.get_SciTail_as_train('/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_train.tsv')
        # seen_classes=[0,2,4,6,8]

        # num_train_optimization_steps = int(
        #     len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank))

    pretrain_model_dir = 'roberta-large-mnli' #'roberta-large' , 'roberta-large-mnli'
    # model = Encoder.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    model = Encoder.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    # exit(0)


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
        train_features = convert_examples_to_features(
            train_examples_entail + train_examples_neutral + train_examples_contra,
            label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        train_features_entail = train_features[:len(train_examples_entail)]
        train_features_neutral = train_features[len(train_examples_entail):(len(train_examples_entail)+len(train_examples_neutral))]
        train_features_contra = train_features[(len(train_examples_entail)+len(train_examples_neutral)):]
        assert len(train_features_entail) == len(train_examples_entail)
        assert len(train_features_neutral) == len(train_examples_neutral)
        assert len(train_features_contra) == len(train_examples_contra)

        '''load 3-shot data'''
        eval_features_shot = convert_examples_to_features(
            train_examples_entail_RTE+train_examples_neutral_RTE + train_examples_contra_RTE,
            label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        eval_all_input_ids_shot = torch.tensor([f.input_ids for f in eval_features_shot], dtype=torch.long)
        eval_all_input_mask_shot = torch.tensor([f.input_mask for f in eval_features_shot], dtype=torch.long)
        eval_all_segment_ids_shot = torch.tensor([f.segment_ids for f in eval_features_shot], dtype=torch.long)
        eval_all_label_ids_shot = torch.tensor([f.label_id for f in eval_features_shot], dtype=torch.long)

        '''load dev set'''
        dev_examples = processor.get_SciTail_as_dev_or_test('/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_dev.tsv', 'dev')
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
        eval_examples = processor.get_SciTail_as_dev_or_test('/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_test.tsv', 'test')
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
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            dataloader_list = []
            for idd, train_features in enumerate([train_features_entail, train_features_neutral, train_features_contra,
            train_features_entail + train_features_neutral + train_features_contra]):
                all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                train_sampler = RandomSampler(train_data)
                '''create 3 samples per class'''
                if idd < 3:
                    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=3)
                else:
                    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                dataloader_list.append(train_dataloader)

            MNLI_entail_dataloader = dataloader_list[0]
            MNLI_neutra_dataloader = dataloader_list[1]
            MNLI_contra_dataloader = dataloader_list[2]
            MNLI_dataloader = dataloader_list[3]

            '''start training'''

            nb_tr_examples, nb_tr_steps = 0, 0
            sample_input_ids_each_iter = []
            sample_input_mask_each_iter = []

            for step, batch in enumerate(tqdm(MNLI_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                assert input_ids.shape[0] == args.train_batch_size

                mnli_entail_batch = get_a_random_batch_from_dataloader(MNLI_entail_dataloader, 3)
                # print('random batch len:', len(mnli_entail_batch[0]))
                mnli_entail_batch_input_ids, mnli_entail_batch_input_mask, mnli_entail_batch_segment_ids, mnli_entail_batch_label_ids = tuple(t.to(device) for t in mnli_entail_batch) #mnli_entail_batch
                # print('sample entail:', mnli_entail_batch_input_ids.shape[0], mnli_entail_batch_label_ids.shape, mnli_entail_batch_label_ids)

                mnli_neutra_batch = get_a_random_batch_from_dataloader(MNLI_neutra_dataloader, 3)
                mnli_neutra_batch_input_ids, mnli_neutra_batch_input_mask, mnli_neutra_batch_segment_ids, mnli_neutra_batch_label_ids = tuple(t.to(device) for t in mnli_neutra_batch) #mnli_neutra_batch
                # print('sample neutra:', mnli_neutra_batch_input_ids.shape[0], mnli_neutra_batch_label_ids.shape, mnli_neutra_batch_label_ids)

                mnli_contra_batch = get_a_random_batch_from_dataloader(MNLI_contra_dataloader, 3)
                mnli_contra_batch_input_ids, mnli_contra_batch_input_mask, mnli_contra_batch_segment_ids, mnli_contra_batch_label_ids = tuple(t.to(device) for t in mnli_contra_batch) #mnli_contra_batch
                # print('sample contra:', mnli_contra_batch_input_ids.shape[0], mnli_contra_batch_label_ids.shape, mnli_contra_batch_label_ids)

                sample_input_ids_i = torch.cat([mnli_entail_batch_input_ids,mnli_neutra_batch_input_ids,mnli_contra_batch_input_ids],dim=0)
                sample_input_ids_each_iter.append(sample_input_ids_i)
                all_input_ids = torch.cat([sample_input_ids_i,input_ids],dim=0)
                assert all_input_ids.shape[0] == args.train_batch_size+9
                sample_input_mask_i = torch.cat([mnli_entail_batch_input_mask,mnli_neutra_batch_input_mask,mnli_contra_batch_input_mask], dim=0)
                sample_input_mask_each_iter.append(sample_input_mask_i)
                all_input_mask = torch.cat([sample_input_mask_i,input_mask], dim=0)

                '''
                forward(self, input_ids, token_type_ids=None, attention_mask=None, sample_size=None, class_size = None, labels=None):
                '''



                '''(1) SciTail samples --> MNLI batch'''
                model.train()
                loss_cross_domain, _ = model(torch.cat([eval_all_input_ids_shot.to(device),input_ids],dim=0), None, torch.cat([eval_all_input_mask_shot.to(device),input_mask], dim=0), sample_size=3, class_size =num_labels, labels=label_ids, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,1,1,1]), prior_samples_outputs=None, is_train=True, loss_fct=loss_fct)
                loss_cross_domain.backward()
                optimizer.step()
                optimizer.zero_grad()
                '''(2) MNLI samples --> MNLI batch'''
                model.train()
                loss, _ = model(all_input_ids, None, all_input_mask, sample_size=3, class_size =num_labels, labels=label_ids, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs=None, is_train=True, loss_fct=loss_fct)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                '''(3) MNLI samples --> SciTail samples'''
                model.train()
                loss_cross_sample, mnli_samples_outputs_i = model(torch.cat([sample_input_ids_i,eval_all_input_ids_shot.to(device)],dim=0), None, torch.cat([sample_input_mask_i,eval_all_input_mask_shot.to(device)], dim=0), sample_size=3, class_size =num_labels, labels=torch.cuda.LongTensor([0,0,0,1,1,1,1,1,1]), sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs=None, is_train=True, loss_fct=loss_fct)
                loss_cross_sample.backward()
                optimizer.step()
                optimizer.zero_grad()




                global_step += 1
                iter_co+=1

                check_freq = 10
                if iter_co %check_freq==0:
                    '''first get info from MNLI by sampling'''
                    assert len(sample_input_ids_each_iter) == check_freq
                    mnli_sample_hidden_list = []
                    mnli_sample_logits_list = []
                    for ff in range(len(sample_input_ids_each_iter)):
                        model.eval()
                        with torch.no_grad():
                            mnli_sample_hidden_i, mnli_sample_logits_i = model(sample_input_ids_each_iter[ff], None, sample_input_mask_each_iter[ff], sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs = None, few_shot_training=False, is_train=False, fetch_hidden_only=True, loss_fct=None)
                            mnli_sample_hidden_list.append(mnli_sample_hidden_i[None,:,:])
                            mnli_sample_logits_list.append(mnli_sample_logits_i[None,:,:])
                    sample_input_ids_each_iter = []

                    sample_input_mask_each_iter = []
                    '''sum or mean does not make big difference'''
                    prior_mnli_samples_outputs = torch.cat(mnli_sample_hidden_list,dim=0)
                    prior_mnli_samples_outputs = torch.mean(prior_mnli_samples_outputs,dim=0)
                    prior_mnli_samples_logits = torch.cat(mnli_sample_logits_list,dim=0)
                    prior_mnli_samples_logits = torch.mean(prior_mnli_samples_logits,dim=0)

                    '''second do few-shot training'''
                    for ff in range(3):
                        model.train()
                        few_loss = model(eval_all_input_ids_shot.to(device), None, eval_all_input_mask_shot.to(device), sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,1,1,1]), prior_samples_outputs = None, few_shot_training=True, is_train=True, loss_fct=loss_fct)
                        few_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        print('few_loss:', few_loss)
                    '''
                    start evaluate on dev set after this epoch
                    '''
                    model.eval()
                    for idd, dev_or_test_dataloader in enumerate([dev_dataloader, eval_dataloader]):
                        logger.info("***** Running evaluation *****")
                        if idd == 0:
                            logger.info("  Num examples = %d", len(dev_examples))
                        else:
                            logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)

                        eval_loss = 0
                        nb_eval_steps = 0
                        preds = []
                        preds_LR= []
                        preds_NN = []
                        gold_label_ids = []
                        print('Evaluating...')
                        for input_ids, input_mask, segment_ids, label_ids in dev_or_test_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)
                            gold_label_ids+=list(label_ids.detach().cpu().numpy())

                            all_input_ids = torch.cat([eval_all_input_ids_shot.to(device),input_ids],dim=0)
                            all_input_mask = torch.cat([eval_all_input_mask_shot.to(device),input_mask], dim=0)


                            with torch.no_grad():
                                logits_LR, logits_NN, logits = model(all_input_ids, None, all_input_mask, sample_size=3, class_size =num_labels, labels=None, sample_labels = torch.cuda.LongTensor([0,0,0,1,1,1,2,2,2]), prior_samples_outputs = prior_mnli_samples_outputs, prior_samples_logits = prior_mnli_samples_logits, is_train=False, loss_fct=None)

                            nb_eval_steps += 1
                            if len(preds) == 0:
                                preds.append(logits.detach().cpu().numpy())
                                preds_LR.append(logits_LR.detach().cpu().numpy())
                                preds_NN.append(logits_NN.detach().cpu().numpy())
                            else:
                                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                                preds_LR[0] = np.append(preds_LR[0], logits_LR.detach().cpu().numpy(), axis=0)
                                preds_NN[0] = np.append(preds_NN[0], logits_NN.detach().cpu().numpy(), axis=0)

                        preds = preds[0]
                        preds_LR = preds_LR[0]
                        preds_NN = preds_NN[0]

                        '''
                        preds: size*3 ["entailment", "neutral", "contradiction"]
                        wenpeng added a softxmax so that each row is a prob vec
                        '''
                        acc_list = []
                        for preds_i in [preds_LR, preds_NN, preds]:
                            pred_probs = softmax(preds_i,axis=1)
                            pred_indices = np.argmax(pred_probs, axis=1)
                            pred_label_ids = []
                            for p in pred_indices:
                                pred_label_ids.append(0 if p == 0 else 1)
                            gold_label_ids = gold_label_ids
                            assert len(pred_label_ids) == len(gold_label_ids)
                            hit_co = 0
                            for k in range(len(pred_label_ids)):
                                if pred_label_ids[k] == gold_label_ids[k]:
                                    hit_co +=1
                            test_acc = hit_co/len(gold_label_ids)

                            acc_list.append(test_acc)


                        softmax_LR = array_2_softmax(preds_LR)
                        softmax_NN = array_2_softmax(preds_NN)
                        preds_ensemble = []
                        for i in range(softmax_LR.shape[0]):
                            if softmax_LR[i][0] > softmax_LR[i][1] and softmax_NN[i][0] > softmax_NN[i][1]:
                                preds_ensemble.append(0)
                            elif softmax_LR[i][0] < softmax_LR[i][1] and softmax_NN[i][0] < softmax_NN[i][1]:
                                preds_ensemble.append(1)
                            elif softmax_LR[i][0] > softmax_LR[i][1] and softmax_LR[i][0] > softmax_NN[i][1]:
                                preds_ensemble.append(0)
                            else:
                                preds_ensemble.append(1)
                        hit_co = 0
                        for k in range(len(preds_ensemble)):
                            if preds_ensemble[k] == gold_label_ids[k]:
                                hit_co +=1
                        test_acc = hit_co/len(gold_label_ids)
                        acc_list.append(test_acc)


                        if idd == 0: # this is dev
                            if np.mean(acc_list) >= max_dev_acc:
                                max_dev_acc = np.mean(acc_list)
                                print('\ndev acc_list:', acc_list, ' max_mean_dev_acc:', max_dev_acc, '\n')
                                '''store the model'''
                                # store_transformers_models(model, tokenizer, '/export/home/Dataset/BERT_pretrained_mine/crossdataentail/trainMNLItestRTE', str(max_dev_acc))

                            else:
                                print('\ndev acc_list:', acc_list, ' max_dev_acc:', max_dev_acc, '\n')
                                break
                        else: # this is test
                            if acc_list[-2] > max_test_acc:
                                max_test_acc = acc_list[-2]
                            print('\ntest acc_list:', acc_list, ' max_test_acc:', max_test_acc, '\n')

def array_2_softmax(a):
    for i in range(a.shape[0]):
        if a[i][2]>a[i][1]:
            a[i][1] = a[i][2]
    sub_a = a[:,:2]
    return softmax(sub_a)

if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=2 python -u train_MNLI_test_3shotSciTail_meta_learning.py --task_name rte --do_train --do_lower_case --bert_model bert-large-uncased --learning_rate 1e-5 --num_train_epochs 3 --data_dir '' --output_dir ''
