
# from load_data import load_BBN_multi_labels_dataset, load_il_groundtruth_as_testset, load_official_testData_il_and_MT, generate_2019_official_output, load_trainingData_types_plus_others,load_trainingData_types,load_SF_type_descriptions, average_f1_two_array_by_col, load_fasttext_multiple_word2vec_given_file, load_word2vec_to_init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import codecs
import torch
import torch.optim as optim
import random
'''head files for using pretrained bert'''
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.optimization import AdamW
# from preprocess_IL3_Uyghur import recover_pytorch_idmatrix_2_text
# from bert_common_functions import sent_to_embedding, sent_to_embedding_last4

'''the following torch seed can result in the same performance'''
torch.manual_seed(400)
device = torch.device("cuda")

from bert_common_functions import sent_pair_to_embedding



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model.eval()
        self.bert_model.to('cuda')

        '''we do not use bias term in representation learning'''
        '''does it have tanh()?'''
        self.label_rep = nn.Linear(768, 2, False) # here we consider three classes: entail, non_entail
        '''we use bias term in classifier'''
        self.classifier = nn.Linear(768, 2)

    def forward(self, sent_pair_batch):
        '''
        sent_pair_batch: a list of list: each sublist has two ele: premise, hypo
        '''
        emb_batch = []
        for sent_pair in sent_pair_batch:
            # print(sent_pair_to_embedding(sent_pair[0], sent_pair[1], self.bert_tokenizer, self.bert_model, False).reshape(1,-1))
            emb_batch.append(sent_pair_to_embedding(sent_pair[0], sent_pair[1], self.bert_tokenizer, self.bert_model, False).reshape(1,-1))
        bert_rep_batch = torch.cat(emb_batch, 0) #(batch, 768)
        batch_scores = (self.label_rep(bert_rep_batch)).tanh()#(batch, 2)
        batch_probs = nn.Softmax(dim=1)((self.classifier(bert_rep_batch)))#(batch, 2)

        return batch_scores, batch_probs

def build_model():
    model = Encoder()
    model.to(device)
    '''binary cross entropy'''
    loss_function = nn.NLLLoss().cuda()
    '''seems weight_decay is not good for LSTM'''
    optimizer = AdamW(model.parameters(), lr=5e-5)#, weight_decay=1e-2)
    return model, loss_function, optimizer


def train_representation_learning(MNLI_pos, MNLI_neg, RTE_pos, RTE_neg, SciTail_pos, SciTail_neg, MNLI_train, MNLI_train_labels, RTE_test, RTE_test_labels, model, loss_function, optimizer):
    # MNLI_pos_len = len(MNLI_pos)
    # MNLI_neg_len = len(MNLI_neg)
    # RTE_pos_len = len(RTE_pos)
    # RTE_neg_len = len(RTE_neg)
    # SciTail_pos_len = len(SciTail_pos)
    # SciTail_neg_len = len(SciTail_neg)
    # sample_size = 10
    # for iter in range(10000000):
    #     model.train()
    #
    #     print('current iter: ', iter)
    #     '''first sample pairs'''
    #     MNLI_pos_sampled_indices = random.sample(range(0,MNLI_pos_len), sample_size)
    #     MNLI_pos_samples = [ MNLI_pos[i] for i in MNLI_pos_sampled_indices]
    #     RTE_pos_sampled_indices = random.sample(range(0,RTE_pos_len), sample_size)
    #     RTE_pos_samples = [ RTE_pos[i] for i in RTE_pos_sampled_indices]
    #     SciTail_pos_sampled_indices = random.sample(range(0,SciTail_pos_len), sample_size)
    #     SciTail_pos_samples = [ SciTail_pos[i] for i in SciTail_pos_sampled_indices]
    #     pos_samples = MNLI_pos_samples+RTE_pos_samples+SciTail_pos_samples
    #
    #     MNLI_neg_sampled_indices = random.sample(range(0,MNLI_neg_len), sample_size)
    #     MNLI_neg_samples = [ MNLI_neg[i] for i in MNLI_neg_sampled_indices]
    #     RTE_neg_sampled_indices = random.sample(range(0,RTE_neg_len), sample_size)
    #     RTE_neg_samples = [ RTE_neg[i] for i in RTE_neg_sampled_indices]
    #     SciTail_neg_sampled_indices = random.sample(range(0,SciTail_neg_len), sample_size)
    #     SciTail_neg_samples = [ SciTail_neg[i] for i in SciTail_neg_sampled_indices]
    #     neg_samples = MNLI_neg_samples + RTE_neg_samples + SciTail_neg_samples
    #     '''we assume 1 means entail, 0 otherwise'''
    #     label_batch = np.array([1]*len(pos_samples)+[0]*len(neg_samples)) # batch
    #     label_batch = autograd.Variable(torch.cuda.LongTensor(label_batch))
    #
    #     model.zero_grad()
    #     batch_scores, _ = model(pos_samples+neg_samples) #(batch, 2)
    #     '''Binary Cross Entropy'''
    #
    #     loss = loss_function(batch_scores, label_batch)
    #     loss.backward()
    #     optimizer.step()
    #     if iter > 50 and iter % 10 == 0:
    #         print('representation learning iter:', iter)
    #         '''now use the pretrained BERT to do classification'''
    train_classifier(MNLI_train, MNLI_train_labels, RTE_test, RTE_test_labels,model, loss_function, optimizer)

def train_classifier(MNLI_train, MNLI_train_labels, RTE_test, RTE_test_labels,model, loss_function, optimizer):
    batch_size =60
    train_groups = len(MNLI_train)//batch_size
    test_group = len(RTE_test)//batch_size
    for i in range(train_groups):
        print('\t\t classifier training group #', i)
        model.train()
        train_batch = MNLI_train[i*batch_size:(i+1)*batch_size]
        train_label_batch = np.array(MNLI_train_labels[i*batch_size:(i+1)*batch_size]) # batch
        train_label_batch = autograd.Variable(torch.cuda.LongTensor(train_label_batch))
        print('train_label_batch:', train_label_batch)
        model.zero_grad()
        _, batch_probs = model(train_batch)
        loss = loss_function(batch_probs, train_label_batch)
        loss.backward()
        optimizer.step()
        if i %10==0:
            '''test on RTE'''
            print('\t\t\t test classifier performace:')
            model.eval()
            pred = []
            with torch.no_grad():
                for j in range(test_group):
                    test_batch = RTE_test[j*batch_size:(j+1)*batch_size]
                    _, batch_probs = model(test_batch)
                    # print('batch_probs:', batch_probs)
                    if len(pred) == 0:
                        pred.append(batch_probs.detach().cpu().numpy())
                    else:
                        pred[0] = np.append(pred[0], batch_probs.detach().cpu().numpy(), axis=0)
            pred = pred[0]
            pred_labels = np.argmax(pred, axis=1)
            print('pred_labels:', pred_labels)
            hit=0
            for k in range(len(pred_labels)):
                if pred_labels[k] == RTE_test_labels[k]:
                    hit+=1
            acc = hit/len(pred_labels)
            print('\t\t\t\t\t\t RTE acc:', acc)










if __name__ == '__main__':
    task_names = ['MNLI', 'GLUE-RTE', 'SciTail']
    # all_entail_training_data = '/export/home/Dataset/MNLI-SNLI-SciTail-RTE-SICK/all.6.train.txt'
    MNLI_pos = []
    MNLI_neg = []
    MNLI_train = []
    MNLI_train_labels = []
    readfile = codecs.open('/export/home/Dataset/glue_data/MNLI/train.tsv', 'r', 'utf-8')
    line_co = 0
    for line in readfile:
        if line_co>0:
            parts=line.strip().split('\t')
            labelstr = parts[-1]
            # MNLI_train.append([parts[8].strip(), parts[9].strip()])
            # MNLI_train_labels.append(1 if labelstr == 'entailment' else 0)
            if labelstr == 'entailment':
                MNLI_pos.append([parts[8].strip(), parts[9].strip()])
                MNLI_train.append([parts[8].strip(), parts[9].strip()])
                MNLI_train_labels.append(1)
            elif labelstr == 'contradiction':
                MNLI_neg.append([parts[8].strip(), parts[9].strip()])
                MNLI_train.append([parts[8].strip(), parts[9].strip()])
                MNLI_train_labels.append(0)
        line_co+=1
    readfile.close()
    print('load MNLI over, sizes: pos', len(MNLI_pos), 'neg', len(MNLI_neg))


    RTE_pos = []
    RTE_neg = []
    RTE_train = []
    RTE_train_labels = []
    readfile = codecs.open('/export/home/Dataset/glue_data/RTE/train.tsv', 'r', 'utf-8')
    line_co = 0
    for line in readfile:
        if line_co>0:
            parts=line.strip().split('\t')
            labelstr = parts[-1]
            RTE_train.append([parts[1].strip(), parts[2].strip()])

            if labelstr == 'entailment':
                RTE_pos.append([parts[1].strip(), parts[2].strip()])
                RTE_train_labels.append(1)
            else:
                RTE_neg.append([parts[1].strip(), parts[2].strip()])
                RTE_train_labels.append(0)
        line_co+=1
    readfile.close()
    print('load RTE over, sizes: pos', len(RTE_pos), 'neg', len(RTE_neg))
    RTE_test = []
    RTE_test_labels = []
    readfile = codecs.open('/export/home/Dataset/RTE/test_RTE_1235.txt', 'r', 'utf-8')
    line_co=0
    for row in readfile:
        line=row.strip().split('\t')
        if len(line)==3:
            RTE_test.append([line[1].strip(), line[2].strip()])
            RTE_test_labels.append(int(line[0])) # 1 means entail, 0 means not-entail

    readfile.close()
    print('loaded test size:', len(RTE_test))



    '''not that scitail has only entails and neutral'''
    SciTail_pos = []
    SciTail_neg = []
    readfile=codecs.open('/export/home/Dataset/SciTailV1/tsv_format/scitail_1.0_train.tsv', 'r', 'utf-8')
    valid=0
    for line in readfile:
        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==3:
            labelstr = parts[2]
            if labelstr == 'neutral':
                SciTail_neg.append([parts[0].strip(), parts[1].strip()])
            elif labelstr == 'entails':
                SciTail_pos.append([parts[0].strip(), parts[1].strip()])
    readfile.close()
    print('load SciTail over, sizes: pos', len(SciTail_pos), 'neg', len(SciTail_neg))

    print("build model...")
    model, loss_function, optimizer = build_model()
    print("training...")
    train_representation_learning(MNLI_pos, MNLI_neg, RTE_pos, RTE_neg, SciTail_pos, SciTail_neg, RTE_train, RTE_train_labels, RTE_test, RTE_test_labels, model, loss_function, optimizer)
