
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

# from bert_common_functions import store_transformers_models, get_a_random_batch_from_dataloader, cosine_rowwise_two_matrices

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
        examples=[]
        # examples_neutral=[]
        # examples_contra=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "train-"+str(line_co-1)
                text_a = line[8].strip()
                text_b = line[9].strip()
                label = line[-1].strip() #["entailment", "neutral", "contradiction"]
                examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
            # if line_co > 1000:
            #     break
        readfile.close()
        print('loaded  size:', line_co)
        return examples




    def get_RTE_as_train(self, filename, K):
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
            sampled_entail = random.sample(entail_list, K)
        else:
            sampled_entail = random.choices(entail_list, k = K)
        if K <= int(not_entail_size/2):
            sampled_not_entail = random.sample(not_entail_list, 2*K)
        else:
            sampled_not_entail = random.choices(not_entail_list, k = 2*K)

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
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_target = RobertaClassificationHead(config)
        self.classifier_target.load_state_dict(self.classifier.state_dict())

    def forward(self, target_id_type_mask, target_labels, source_id_type_mask, source_labels, loss_fct=None):

        '''
        samples: input_ids, token_type_ids, attention_mask; in class order
        minibatch: input_ids, token_type_ids, attention_mask
        '''
        '''k-example in test'''
        target_input_ids, target_token_type, target_attention_mask = target_id_type_mask
        target_input_size = target_input_ids.shape[0]
        '''mnli minibatch'''
        if source_id_type_mask is not None:
            source_input_ids, source_token_type, source_attention_mask = source_id_type_mask

            input_ids = torch.cat([target_input_ids, source_input_ids], dim=0)
            # token_type_ids =torch.cat([target_token_type, source_token_type], dim=0)
            attention_mask = torch.cat([target_attention_mask, source_attention_mask], dim=0)
        else:
            input_ids = target_input_ids
            # token_type_ids = target_token_type
            attention_mask = target_attention_mask

        '''pls note that roberta does not need token_type, especially when value more than 0 in the tensor, error report'''
        outputs = self.roberta(input_ids, attention_mask, None)
        pooled_outputs = outputs[1]#torch.max(outputs[0],dim=1)[0]+ outputs[1]#outputs[1]#torch.mean(outputs[0],dim=1)#outputs[1] #(batch, hidden_size)
        '''mnli minibatch'''
        if source_id_type_mask is not None:
            LR_logits_source = self.classifier(pooled_outputs[target_input_size:]) #(9+batch, 3)
        '''target (k) examples'''
        LR_logits_target = (self.classifier_target(pooled_outputs[:target_input_size])+
            self.classifier(pooled_outputs[:target_input_size]))

        target_loss = loss_fct(LR_logits_target.view(-1, self.num_labels), target_labels.view(-1))
        if source_id_type_mask is not None:
            source_loss = loss_fct(LR_logits_source.view(-1, self.num_labels), source_labels.view(-1))
            loss = target_loss+source_loss
        else:
            '''testing, compute acc'''
            pred_labels_batch = torch.softmax(LR_logits_target.view(-1, self.num_labels), dim=1).argmax(dim=1)
            pred_labels_batch[pred_labels_batch!=0]=1
            gold_labels_batch = target_labels
            gold_labels_batch[gold_labels_batch!=0]=1
            acc = (pred_labels_batch == gold_labels_batch).sum().float() / float(target_labels.size(0) )
            loss = acc
        return loss


class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

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
    parser.add_argument("--k_shot",
                        default=3,
                        type=int,
                        required=True,
                        help="size per class")
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
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
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



    train_examples_source = processor.get_MNLI_as_train('/export/home/Dataset/glue_data/MNLI/train.tsv') #train_pu_half_v1.txt
    '''load k-shot examples'''
    train_examples_entail_RTE, train_examples_neutral_RTE, train_examples_contra_RTE = processor.get_RTE_as_train('/export/home/Dataset/glue_data/RTE/train.tsv', args.k_shot)
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
    # print('param_optimizer:', param_optimizer)
    # exit(0)
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
        train_source_features = convert_examples_to_features(
            train_examples_source,
            label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        train_source_input_ids = torch.tensor([f.input_ids for f in train_source_features], dtype=torch.long)
        train_source_input_mask = torch.tensor([f.input_mask for f in train_source_features], dtype=torch.long)
        train_source_segment_ids = torch.tensor([f.segment_ids for f in train_source_features], dtype=torch.long)
        train_source_label_ids = torch.tensor([f.label_id for f in train_source_features], dtype=torch.long)

        train_source_data = TensorDataset(train_source_input_ids, train_source_input_mask, train_source_segment_ids, train_source_label_ids)
        train_source_sampler = RandomSampler(train_source_data)
        '''create 3 samples per class'''
        train_source_dataloader = DataLoader(train_source_data, sampler=train_source_sampler, batch_size=args.train_batch_size)


        '''load 3-shot data'''
        train_target_features = convert_examples_to_features(
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

        train_target_input_ids_shot = torch.tensor([f.input_ids for f in train_target_features], dtype=torch.long).to(device)
        train_target_input_mask_shot = torch.tensor([f.input_mask for f in train_target_features], dtype=torch.long).to(device)
        train_target_segment_ids_shot = torch.tensor([f.segment_ids for f in train_target_features], dtype=torch.long).to(device)
        train_target_label_ids_shot = torch.tensor([f.label_id for f in train_target_features], dtype=torch.long).to(device)
        assert train_target_input_ids_shot.shape[0] == args.k_shot*3

        train_target_data = TensorDataset(train_target_input_ids_shot, train_target_input_mask_shot, train_target_segment_ids_shot, train_target_label_ids_shot)
        train_target_sampler = RandomSampler(train_target_data)
        '''create 3 samples per class'''
        assert args.k_shot*3>=9
        train_target_dataloader = DataLoader(train_target_data, sampler=train_target_sampler, batch_size=9)


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
        test_examples = processor.get_RTE_as_test('/export/home/Dataset/RTE/test_RTE_1235.txt')
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running training *****")
        iter_co = 0
        tr_loss = 0
        loss_fct = CrossEntropyLoss()
        max_dev_acc = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):

            '''
            loop on the k-shot examples
            '''
            for train_target_batch in train_target_dataloader:
                train_target_batch = tuple(t.to(device) for t in train_target_batch)
                train_target_input_ids_batch, train_target_input_mask_batch, train_target_segment_ids_batch, train_target_label_ids_batch = train_target_batch


                target_id_type_mask_batch = (train_target_input_ids_batch, train_target_segment_ids_batch, train_target_input_mask_batch)
                # target_id_type_mask_batch = (train_target_input_ids_shot, None, train_target_input_mask_shot)
                target_labels_batch = train_target_label_ids_batch

                # for step, train_source_batch in enumerate(tqdm(train_source_dataloader, desc="Iteration")):
                for train_source_batch in train_source_dataloader:
                    '''we make sure one scan of train_target_dataloader corresponds to one scan of train_source_dataloader'''
                    rand_prob = random.uniform(0, 1)
                    if rand_prob > 1/len(train_target_dataloader):
                        continue

                    train_source_batch = tuple(t.to(device) for t in train_source_batch)
                    train_source_input_ids_batch, train_source_input_mask_batch, train_source_segment_ids_batch, train_source_label_ids_batch = train_source_batch

                    # assert train_source_input_ids_batch.shape[0] == args.train_batch_size



                    source_id_type_mask_batch = (train_source_input_ids_batch, train_source_segment_ids_batch, train_source_input_mask_batch)
                    # source_id_type_mask_batch = (train_source_input_ids_batch, None, train_source_input_mask_batch)
                    source_labels_batch = train_source_label_ids_batch

                    model.train()
                    loss_cross_domain = model(target_id_type_mask_batch, target_labels_batch, source_id_type_mask_batch, source_labels_batch, loss_fct=loss_fct)
                    loss_cross_domain.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                    global_step += 1
                    iter_co+=1


                    if iter_co %50==0:
                        model.eval()
                        dev_acc = 0.0
                        with torch.no_grad():
                            for idd, target_dev_batch in enumerate(dev_dataloader):

                                target_dev_batch = tuple(t.to(device) for t in target_dev_batch)
                                target_dev_input_ids_batch, target_dev_input_mask_batch, target_dev_segment_ids_batch, target_dev_label_ids_batch = target_dev_batch

                                target_id_type_mask_batch = (target_dev_input_ids_batch, target_dev_segment_ids_batch, target_dev_input_mask_batch)
                                target_labels_batch = target_dev_label_ids_batch

                                acc_i = model(target_id_type_mask_batch, target_labels_batch, None, None, loss_fct=loss_fct)
                                dev_acc+=acc_i.item()

                        dev_acc/=len(dev_dataloader)
                        print('iter:', iter_co, ' dev acc:', dev_acc)
                        if dev_acc> max_dev_acc:
                            max_dev_acc = dev_acc
                            print('max_dev_acc:', max_dev_acc)
                            '''testing'''
                            test_acc = 0.0
                            with torch.no_grad():
                                for idd, target_test_batch in enumerate(test_dataloader):

                                    target_test_batch = tuple(t.to(device) for t in target_test_batch)
                                    target_test_input_ids_batch, target_test_input_mask_batch, target_test_segment_ids_batch, target_test_label_ids_batch = target_test_batch

                                    target_id_type_mask_batch = (target_test_input_ids_batch, target_test_segment_ids_batch, target_test_input_mask_batch)
                                    target_labels_batch = target_test_label_ids_batch

                                    acc_i = model(target_id_type_mask_batch, target_labels_batch, None, None, loss_fct=loss_fct)
                                    test_acc+=acc_i.item()

                            test_acc/=len(test_dataloader)
                            print('\t\t\t >>>>test acc:', test_acc)
if __name__ == "__main__":
    main()
    '''
    1, change the encoder to the full roberta-large-mnli
    2, change the k-shot size easily
    '''
# CUDA_VISIBLE_DEVICES=0 python -u forfun.py --task_name rte --do_train --do_lower_case --bert_model bert-large-uncased --learning_rate 1e-5 --data_dir '' --output_dir '' --k_shot 3
