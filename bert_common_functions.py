import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
import os

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
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    last_layer_output = encoded_layers[-1]
    return last_layer_output[0][0]

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
    output_config_file = os.path.join(output_dir, 'bert_config.json')

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
