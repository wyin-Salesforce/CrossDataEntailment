import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pytorch_transformers.modeling_bert import BertPreTrainedModel

class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

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
