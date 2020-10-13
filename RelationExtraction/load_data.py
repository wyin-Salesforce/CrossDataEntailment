import json
import codecs
import random
from collections import defaultdict



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def load_FewRel_data(k_shot):
    '''first load relation definicatinos'''
    relation_2_desc = {}
    with open('/export/home/Dataset/FewRel.1.0/pid2name.json') as json_file:
        relation_data = json.load(json_file)
        # print(len(relation_data.keys()))
        for relation, definition in relation_data.items():
            assert len(definition) == 2
            # print('definition:', definition)
            relation_2_desc[relation] = definition


    dev_relation_2_examples = {}
    filenames = ['train_wiki.json', 'val_wiki.json']
    for filename in filenames:
        with open('/export/home/Dataset/FewRel.1.0/'+filename) as json_file:
            dev_data = json.load(json_file)
            for relation, example_list in dev_data.items():
                assert len(example_list) == 700
                tup_list = []
                for example in example_list:
                    sent = ' '.join(example.get('tokens'))
                    head_entity = example.get('h')[0]
                    tail_entity = example.get('t')[0]
                    tup_list.append((sent, head_entity, tail_entity))
                dev_relation_2_examples[relation] = tup_list
    json_file.close()
    assert len(dev_relation_2_examples.keys()) == 80
    dev_4_train = {}
    dev_4_dev = {}
    dev_4_test = {}
    '''50, 5, 10'''
    for relation, ex_list in dev_relation_2_examples.items():
        dev_4_train[relation] = ex_list[:50]
        dev_4_dev[relation] = ex_list[50:55]
        dev_4_test[relation] = ex_list[55:65]

    selected_dev_4_train = {}
    for relation, ex_list in dev_4_train.items():
        selected_dev_4_train[relation] = ex_list if k_shot == 0 else random.sample(ex_list, k_shot)

    '''build train'''
    train_examples = []
    ex_id = 0
    for relation, example_list in selected_dev_4_train.items():
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            train_examples.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            for relation_i in selected_dev_4_train.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    train_examples.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1

    '''build dev'''
    dev_examples = []
    ex_id = 0
    for relation, example_list in dev_4_dev.items():
        assert len(example_list) == 5
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            dev_examples.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            # print('dev_4_dev.keys():', len(dev_4_dev.keys()))
            for relation_i in dev_4_dev.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    dev_examples.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1
    # print('dev_examples:', len(dev_examples))
    '''build test'''
    test_examples = []
    ex_id = 0
    for relation, example_list in dev_4_test.items():
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            test_examples.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            for relation_i in dev_4_test.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    test_examples.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1


    return train_examples, dev_examples, test_examples


def load_FewRel_GFS_Entail(k_shot):
    '''first load relation definicatinos'''
    relation_2_desc = {}
    with open('/export/home/Dataset/FewRel.1.0/pid2name.json') as json_file:
        relation_data = json.load(json_file)
        # print(len(relation_data.keys()))
        for relation, definition in relation_data.items():
            assert len(definition) == 2
            # print('definition:', definition)
            relation_2_desc[relation] = definition


    dev_relation_2_examples = {}
    filenames = ['train_wiki.json', 'val_wiki.json']
    for filename in filenames:
        with open('/export/home/Dataset/FewRel.1.0/'+filename) as json_file:
            dev_data = json.load(json_file)
            for relation, example_list in dev_data.items():
                assert len(example_list) == 700
                tup_list = []
                for example in example_list:
                    sent = ' '.join(example.get('tokens'))
                    head_entity = example.get('h')[0]
                    tail_entity = example.get('t')[0]
                    tup_list.append((sent, head_entity, tail_entity))
                dev_relation_2_examples[relation] = tup_list
    json_file.close()
    assert len(dev_relation_2_examples.keys()) == 80
    dev_4_train = {}
    dev_4_dev = {}
    dev_4_test = {}
    '''50, 5, 10'''
    for relation, ex_list in dev_relation_2_examples.items():
        dev_4_train[relation] = ex_list[:50]
        dev_4_dev[relation] = ex_list[50:55]
        dev_4_test[relation] = ex_list[55:65]

    selected_dev_4_train = {}
    for relation, ex_list in dev_4_train.items():
        selected_dev_4_train[relation] = ex_list if k_shot == 0 else random.sample(ex_list, k_shot)

    '''build train'''
    train_examples_entail = []
    train_examples_nonentail = []
    ex_id = 0
    for relation, example_list in selected_dev_4_train.items():
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            train_examples_entail.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            for relation_i in selected_dev_4_train.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    train_examples_nonentail.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1

    '''build dev'''
    dev_examples = []
    ex_id = 0
    for relation, example_list in dev_4_dev.items():
        # assert len(example_list) == 1
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            dev_examples.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            # print('dev_4_dev.keys():', len(dev_4_dev.keys()))
            for relation_i in dev_4_dev.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    dev_examples.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1
    # print('dev_examples:', len(dev_examples))
    '''build test'''
    test_examples = []
    ex_id = 0
    for relation, example_list in dev_4_test.items():
        relation_desc = relation_2_desc.get(relation)
        for example in example_list:
            sentence, head_ent, tail_ent = example
            '''positive hypo'''
            hypo = head_ent+' is '+relation_desc[0]+' of '+tail_ent
            test_examples.append(
                InputExample(guid=ex_id, text_a=sentence, text_b=hypo, label='entailment'))
            '''negative hypo'''
            for relation_i in dev_4_test.keys():
                if relation_i != relation:
                    relation_i_desc = relation_2_desc.get(relation_i)
                    hypo_neg = head_ent+' is '+relation_i_desc[0]+' of '+tail_ent
                    test_examples.append(
                        InputExample(guid=ex_id, text_a=sentence, text_b=hypo_neg, label='non_entailment'))
            ex_id+=1


    return train_examples_entail, train_examples_nonentail, dev_examples, test_examples


if __name__ == "__main__":
    load_FewRel_data(10)


'''
80 classes; (50, 5, 10)

50*6400 = 320000
pos: 32000
neg:
'''
