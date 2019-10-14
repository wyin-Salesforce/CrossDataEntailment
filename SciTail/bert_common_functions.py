import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from random import randrange

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

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def sent_pair_to_embedding(sent1, sent2, tokenizer, model, tokenized_yes):
    if tokenized_yes:
        sent1_tokenized = sent1.split()
        sent2_tokenized = sent2.split()
    else:
        sent1_tokenized = tokenizer.tokenize(sent1)
        sent2_tokenized = tokenizer.tokenize(sent2)
    pair_tokenlist = ['[CLS]']+sent1_tokenized+['[SEP]']+sent2_tokenized+['[SEP]']
    # print(pair_tokenlist)
    segments_ids = [0]*(len(sent1_tokenized)+2)+[1]*(len(sent2_tokenized)+1)
    # print(segments_ids)
    indexed_tokens = tokenizer.convert_tokens_to_ids(pair_tokenlist)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')


    # Predict hidden states features for each layer
    with torch.no_grad():
        last_hidden_states = model(tokens_tensor, segments_tensors)[0]

    return last_hidden_states[0,0,:]

def sent_to_embedding(sent1, tokenizer, model, tokenized_yes):
    sent1_tokenized = tokenizer.tokenize(sent1)


    pair_tokenlist = ["[CLS]"] + sent1_tokenized + ["[SEP]"]
    segments_ids = [0] * len(pair_tokenlist)

    indexed_tokens = tokenizer.convert_tokens_to_ids(pair_tokenlist)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor[:,:512], segments_tensors[:,:512])

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    last_layer_output = encoded_layers[-1]
    return last_layer_output[0][0]

def sent_to_embedding_last4(sent1, tokenizer, model, tokenized_yes):
    sent1_tokenized = tokenizer.tokenize(sent1)


    pair_tokenlist = ["[CLS]"] + sent1_tokenized + ["[SEP]"]
    segments_ids = [0] * len(pair_tokenlist)

    indexed_tokens = tokenizer.convert_tokens_to_ids(pair_tokenlist)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor[:,:512], segments_tensors[:,:512])

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    last_layer_output = encoded_layers[-1] #()
    first_layer_output = encoded_layers[-4] #()
    second_layer_output = encoded_layers[-3] #()
    third_layer_output = encoded_layers[-2] #()

    return torch.cat([first_layer_output[0][0],second_layer_output[0][0],third_layer_output[0][0],last_layer_output[0][0]], 0)

def sent_to_embedding_matrix(sent1, tokenizer, model, tokenized_yes, max_len):
    '''
    we get contextualized token-level representations
    '''
    sent1_tokenized = tokenizer.tokenize(sent1)


    pair_tokenlist = ["[CLS]"] + sent1_tokenized + ["[SEP]"]
    segments_ids = [0] * len(pair_tokenlist)

    indexed_tokens = tokenizer.convert_tokens_to_ids(pair_tokenlist)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor[:,:512], segments_tensors[:,:512])

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    last_layer_output = encoded_layers[-1] #(1, len, hidden_size)
    # first_layer_output = encoded_layers[-4] #()
    # second_layer_output = encoded_layers[-3] #()
    # third_layer_output = encoded_layers[-2] #()
    '''we do not need the hidden states of the [CLS] and [SEP]'''
    origin_matrix = last_layer_output[0][:-1]
    append_size = max_len - origin_matrix.size(0)
    if append_size>0:
        append_matrix =  torch.repeat_interleave(origin_matrix[-1:], append_size, dim=0) #(append_size, hidden_size)
        return torch.cat([origin_matrix, append_matrix],0)
    else:
        return origin_matrix[:max_len]

class LogisticRegression(nn.Module):  # inheriting from nn.Module!

    def __init__(self, feature_size, label_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(feature_size, label_size)

    def forward(self, feature_vec):
        return F.log_softmax(self.linear(feature_vec), dim=1)


def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('starting model storing....')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('store succeed')

def get_a_random_batch_from_dataloader(dataloader, size):
    while True:
        ith = randrange(len(dataloader))
        for step, batch in enumerate(dataloader):
            if step == ith:
                if len(batch[0]) ==  size:
                    return batch
                else:
                    break
def cosine_rowwise_two_matrices(a,b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    row_size = a_norm.shape[0]
    col_size = a_norm.shape[1]
    return torch.bmm(a_norm.view(row_size, 1, col_size), b_norm.view(row_size, col_size, 1)).view(row_size, -1)


def store_bert_model(model, vocab, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('starting model storing....')
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
    output_config_file = os.path.join(output_dir, 'config.json')

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)

    """Save the tokenizer vocabulary to a directory or file."""
    index = 0
    if os.path.isdir(output_dir):
        vocab_file = os.path.join(output_dir, 'vocab.txt')
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                               " Please check that the vocabulary is not corrupted!".format(vocab_file))
                index = token_index
            writer.write(token + u'\n')
            index += 1
    writer.close()
    print('store succeed')
