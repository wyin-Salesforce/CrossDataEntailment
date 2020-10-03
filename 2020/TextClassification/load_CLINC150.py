import json
import codecs
import random
from collections import defaultdict
domain2intents={'banking':['transfer','transactions','balance','freeze account',
'pay bill', 'bill balance', 'bill due', 'interest rate', 'routing number',
'minimum payment', 'order checks', 'pin change', 'report fraud', 'account blocked',
'spending history'],
'credit cards':['credit score', 'report lost card', 'credit limit', 'rewards balance',
'new card', 'application status', 'card declined', 'international fees', 'apr',
'redeem rewards', 'credit limit change', 'damaged card', 'replacement card duration',
'improve credit score', 'expiration date'],
'kitchen & dining':['recipe','restaurant reviews','calories','nutrition info',
'restaurant suggestion', 'ingredients list', 'ingredient substitution', 'cook time',
'food last', 'meal suggestion', 'restaurant reservation', 'confirm reservation',
'how busy', 'cancel reservation','accept reservation'],
'home':['shopping list', 'shopping list update', 'next song', 'play music', 'update playlist',
'todo list', 'todo list update', 'calendar', 'calendar update', 'what song', 'order',
'order status', 'reminder', 'reminder update', 'smart home'],
'auto & commute':['traffic', 'directions', 'gas', 'gas type', 'distance', 'current location',
'mpg', 'oil change when','oil change how', 'jump start', 'uber', 'schedule maintenance',
'last maintenance', 'tire pressure', 'tire change'],
'travel':['book flight', 'book hotel', 'car rental', 'travel suggestion', 'travel alert',
'travel notification', 'carry on', 'timezone', 'vaccines', 'translate', 'flight status',
'international visa', 'lost luggage', 'plug type', 'exchange rate'],
'utility':['time', 'alarm', 'share location', 'find phone', 'weather', 'text', 'spelling', 'make call',
'timer', 'date', 'calculator', 'measurement conversion', 'flip coin', 'roll dice', 'definition'],
'work':['direct deposit', 'pto request', 'taxes', 'payday', 'w2', 'pto balance', 'pto request status',
'next holiday', 'insurance', 'insurance change', 'schedule meeting', 'pto used', 'meeting schedule',
'rollover 401k', 'income'],
'small talk':['greeting', 'goodbye', 'tell joke', 'where are you from', 'how old are you', 'what is your name',
'who made you', 'thank you', 'what can I ask you', 'what are your hobbies', 'do you have pets', 'are you a bot',
'meaning of life', 'who do you work for', 'fun fact'],
'meta':['change AI name', 'change user name', 'cancel', 'user name', 'reset settings', 'whisper mode',
'repeat', 'no', 'yes', 'maybe', 'change language', 'change accent', 'change volume', 'change speed',
'sync device']}

dataIntent_2_realIntent={'routing':'routing number', 'what can i ask you':'what can I ask you', 'change ai name':'change AI name',
'min payment':'minimum payment', 'accept reservations': 'accept reservation'}



all_seen_intents = []
for domain, intent_list in domain2intents.items():
    all_seen_intents+=intent_list
all_seen_intent_set = set(all_seen_intents)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def load_CLINC150_full(filename, k_shot):

    readfile = codecs.open(filename, 'r', 'utf-8')
    file2dict =  json.load(readfile)
    intent_set = set()
    train_intent2examples={}
    dev_intent2examples={}
    test_intent2examples={}

    interested_intents = domain2intents.get('banking')
    for key, value in file2dict.items():
        print(key, len(value))
        if key in set(['train', 'val','test']):
            for sub_list in value:
                sentence = sub_list[0].strip()
                # length2size[len(sentence.split())]+=1
                intent = ' '.join(sub_list[1].split('_'))
                intent = dataIntent_2_realIntent.get(intent, intent)
                if intent in set(interested_intents):
                    '''this intent is in the target domain'''
                    if key == 'train':
                        examples = train_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        train_intent2examples[intent] = examples
                    elif key == 'val':
                        examples = dev_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        dev_intent2examples[intent] = examples
                    else:
                        examples = test_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        test_intent2examples[intent] = examples
        elif key in set(['oos_val','oos_test']):
            for sub_list in value:
                sentence = sub_list[0].strip()
                # length2size[len(sentence.split())]+=1
                intent = 'oos'
                if key == 'oos_val':
                    examples = dev_intent2examples.get(intent)
                    if examples is None:
                        examples = []
                    examples.append(sentence)
                    dev_intent2examples[intent] = examples
                else:
                    examples = test_intent2examples.get(intent)
                    if examples is None:
                        examples = []
                    examples.append(sentence)
                    test_intent2examples[intent] = examples

    # print('length2size:', length2size)
    '''confirm everything is correct'''
    assert len(train_intent2examples.keys()) == 15
    assert len(dev_intent2examples.keys()) == 15+1
    assert len(test_intent2examples.keys()) == 15+1
    for key, valuelist in train_intent2examples.items():
        assert len(valuelist) == 100
    for key, valuelist in dev_intent2examples.items():
        if key == 'oos':
            assert len(valuelist) == 100
        else:
            assert len(valuelist) == 20
    for key, valuelist in test_intent2examples.items():
        if key == 'oos':
            assert len(valuelist) == 1000
        else:
            assert len(valuelist) == 30

    '''k-shot sampling'''
    '''train'''
    sampled_train_intent2examples={}
    for intent, example_list in train_intent2examples.items():
        sampled_examples = random.sample(example_list, k_shot)
        sampled_train_intent2examples[intent]=sampled_examples
    train_examples = []
    ex_id = 0
    for intent, example_list in sampled_train_intent2examples.items():
        for example in example_list:
            '''positive hypo'''
            train_examples.append(
                InputExample(guid=ex_id, text_a=example, text_b=intent, label='entailment'))
            '''negative hypo'''
            for intent_i in sampled_train_intent2examples.keys():
                if intent_i != intent:
                    train_examples.append(
                        InputExample(guid=ex_id, text_a=example, text_b=intent_i, label='non_entailment'))
            ex_id+=1
    '''dev'''
    dev_examples = []
    ex_id = 0
    for intent, example_list in dev_intent2examples.items():
        for example in example_list:
            '''positive hypo'''
            if intent != 'oos':
                dev_examples.append(
                    InputExample(guid=ex_id, text_a=example, text_b=intent, label='entailment'))
            '''negative hypo'''
            for intent_i in dev_intent2examples.keys():
                if intent_i != intent:
                    dev_examples.append(
                        InputExample(guid=ex_id, text_a=example, text_b=intent_i, label='non_entailment'))
            ex_id+=1
    '''test'''
    test_examples = []
    ex_id = 0
    for intent, example_list in test_intent2examples.items():
        for example in example_list:
            '''positive hypo'''
            if intent != 'oos':
                test_examples.append(
                    InputExample(guid=ex_id, text_a=example, text_b=intent, label='entailment'))
            '''negative hypo'''
            for intent_i in test_intent2examples.keys():
                if intent_i != intent:
                    test_examples.append(
                        InputExample(guid=ex_id, text_a=example, text_b=intent_i, label='non_entailment'))
            ex_id+=1

    print('train pair size:', len(train_examples), ' dev pair size:', len(dev_examples), ', test pair size:', len(test_examples))
    return train_examples, dev_examples, test_examples


def load_OOS():

    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')
    file2dict =  json.load(readfile)
    dev_intent2examples={}
    test_intent2examples={}

    for key, value in file2dict.items():
        if key in set(['oos_val','oos_test']):
            for sub_list in value:
                sentence = sub_list[0].strip()
                intent = 'oos'
                if key == 'oos_val':
                    examples = dev_intent2examples.get(intent)
                    if examples is None:
                        examples = []
                    examples.append(sentence)
                    dev_intent2examples[intent] = examples
                else:
                    examples = test_intent2examples.get(intent)
                    if examples is None:
                        examples = []
                    examples.append(sentence)
                    test_intent2examples[intent] = examples
    '''dev'''
    dev_examples = []
    for intent, example_list in dev_intent2examples.items():
        sampled_examples = example_list#random.sample(example_list, k)
        for example in sampled_examples:
            dev_examples.append(
                InputExample(guid='dev_ex', text_a=example, text_b=None, label=intent))
    '''test'''
    test_examples = []
    for intent, example_list in test_intent2examples.items():
        sampled_examples = example_list#random.sample(example_list, k)
        for example in sampled_examples:
            test_examples.append(
                InputExample(guid='test_ex', text_a=example, text_b=None, label=intent))
    print('size:', len(dev_examples), len(test_examples))
    return dev_examples, test_examples


def load_FewRel_dev():
    with open('/export/home/Dataset/FewRel.1.0/val_wiki.json') as json_file:
        dev_data = json.load(json_file)
        print(len(dev_data.keys()), dev_data.keys())




if __name__ == "__main__":
    # load_CLINC150()
    load_FewRel_dev()
