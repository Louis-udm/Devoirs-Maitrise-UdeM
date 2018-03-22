#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# model-bi-gramms.py
# @author Zhibin.LU
# @created Fri Feb 23 2018 17:14:32 GMT-0500 (EST)
# @last-modified Wed Mar 14 2018 19:11:45 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# # # #

import gzip
import time
from collections import Counter

import regex as re
import spacy
import textacy
import loader


def load_data(folder):
    """
    Load text in a string.
    """
    file_paths = loader.list_files(folder)

    input_words = []
    target_words = []

    for file_path in file_paths:
        with gzip.open(file_path, 'rt', encoding='ISO-8859-1') as f:
            lines = f.read().split('\n')

        for line in lines:
            if line.startswith('#begin') or line.startswith('#end'):
                continue
            line = line.encode("ascii", errors="ignore").decode()

            split_result = line.split('\t')
            if len(split_result) == 2:
                target_word, input_word = split_result
                input_word = input_word.lower().strip()
                target_word = target_word.lower().strip()
                pattern = re.compile(r'\'')
                input_word = re.sub(pattern, '', input_word)
                target_word = re.sub(pattern, '', target_word)

                input_word = re.sub("([\?\!\~\&\=\[\]\{\}\<\>\(\)\_\-\+\/\.])", r" \1 ", input_word)
                target_word = re.sub("([\?\!\~\&\=\[\]\{\}\<\>\(\)\_\-\+\/\.])", r" \1 ", target_word)

                pattern = re.compile(r'\d+s')
                m1 = re.search(pattern, input_word)
                m2 = re.search(pattern, target_word)
                if m2 is not None and m1 is None:
                    input_word = re.sub('(\d+)', r"\1s", input_word)

                input_word = re.sub('(\d+)', r" \1 ", input_word)
                target_word = re.sub('(\d+)', r" \1 ", target_word)

                input_word = re.sub(' +', ' ', input_word)
                target_word = re.sub(' +', ' ', target_word)
                if input_word == '':
                    continue
                input_words.append(input_word)
                target_words.append(target_word)

    return ' '.join(input_words), ' '.join(target_words)


print("{} Loading data...".format(time.strftime("%d-%m-%Y %H:%M:%S")))
train_lemm_corpus, train_surf_corpus = load_data('data/train')
test_lemm_corpus, test_surf_corpus = load_data('data/test')
train_lemm_corpus = re.sub(' +', ' ', train_lemm_corpus)
train_surf_corpus = re.sub(' +', ' ', train_surf_corpus)
test_lemm_corpus = re.sub(' +', ' ', test_lemm_corpus)
test_surf_corpus = re.sub(' +', ' ', test_surf_corpus)

# %%
'''
Get 2-gramms model, all types, all sentences of train_lemme set.
Get 2-gramms model, all types, all sentences of train_surface set.
Get all types, all sentences of test_lemme set.
Get all types, all sentences of test_surface set.
'''

print("{} Training model...".format(time.strftime("%d-%m-%Y %H:%M:%S")))
start_time = time.time()

nlp = spacy.load('en', disable=['parser', 'tagger'])
train_lemm_tacy_doc = nlp(train_lemm_corpus)
train_surf_tacy_doc = nlp(train_surf_corpus)
test_lemm_tacy_doc = nlp(test_lemm_corpus)
test_surf_tacy_doc = nlp(test_surf_corpus)

print('Tokens of train_lemm_tacy_doc: ', len(train_lemm_tacy_doc))
print('Tokens of train_surf_tacy_doc: ', len(train_surf_tacy_doc))
if len(train_lemm_tacy_doc) != len(train_surf_tacy_doc):
    print('Warning: the numbre of tokens of lemme and surfaceis in train not equal !!!!!!')

print('Tokens of test_lemm_tacy_doc: ', len(test_lemm_tacy_doc))
print('Tokens of test_surf_tacy_doc: ', len(test_surf_tacy_doc))
if len(test_lemm_tacy_doc) != len(test_surf_tacy_doc):
    print('Warning: the numbre of tokens of lemme and surfaceis on test not equal !!!!!!')

# %%
train_surf_tacy_sents = []
start_ind = 0
for token in train_surf_tacy_doc:
    if token.text in ['.', '?', '!']:
        train_surf_tacy_sents.append(train_surf_tacy_doc[start_ind:token.i + 1])
        start_ind = token.i + 1
print('total sentence of train surf:', len(train_surf_tacy_sents))
train_lemm_tacy_sents = []
start_ind = 0
for token in train_lemm_tacy_doc:
    if token.text in ['.', '?', '!']:
        train_lemm_tacy_sents.append(train_lemm_tacy_doc[start_ind:token.i + 1])
        start_ind = token.i + 1
print('total sentence of train lemm:', len(train_lemm_tacy_sents))

if len(train_surf_tacy_sents) != len(train_lemm_tacy_sents):
    print('Warning: the numbre of sentances of lemme and surface is not equal !!!!!!')

test_surf_tacy_sents = []
start_ind = 0
for token in test_surf_tacy_doc:
    if token.text in ['.', '?', '!']:
        test_surf_tacy_sents.append(test_surf_tacy_doc[start_ind:token.i + 1])
        start_ind = token.i + 1
print('total sentence of test surf:', len(test_surf_tacy_sents))
test_lemm_tacy_sents = []
start_ind = 0
for token in test_lemm_tacy_doc:
    if token.text in ['.', '?', '!']:
        test_lemm_tacy_sents.append(test_lemm_tacy_doc[start_ind:token.i + 1])
        start_ind = token.i + 1
print('total sentence of test lemm:', len(test_lemm_tacy_sents))

if len(test_surf_tacy_sents) != len(test_lemm_tacy_sents):
    print('Warning: the numbre of sentances of lemme and surface on test is not equal !!!!!!')

# %%
train_lemm_tacy_doc = textacy.Doc(train_lemm_tacy_doc)
train_surf_tacy_doc = textacy.Doc(train_surf_tacy_doc)

train_lemm_2grams_bag = train_lemm_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower', named_entities=False,
                                                            weighting='count', as_strings=True, filter_stops=False,
                                                            filter_punct=False, filter_nums=False,
                                                            drop_determiners=False)
print('size of train lemm 2grams bag:', len(train_lemm_2grams_bag))
train_lemm_1grams_bag = train_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower', named_entities=False,
                                                            weighting='count', as_strings=True, filter_stops=False,
                                                            filter_punct=False, filter_nums=False,
                                                            drop_determiners=False)
print('size of train lemm 1grams bag:', len(train_lemm_1grams_bag))

train_surf_2grams_bag = train_surf_tacy_doc.to_bag_of_terms(ngrams=2, normalize='lower', named_entities=False,
                                                            weighting='count', as_strings=True, filter_stops=False,
                                                            filter_punct=False, filter_nums=False,
                                                            drop_determiners=False)
print('size of train surf 2grams bag:', len(train_surf_2grams_bag))
train_surf_1grams_bag = train_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower', named_entities=False,
                                                            weighting='count', as_strings=True, filter_stops=False,
                                                            filter_punct=False, filter_nums=False,
                                                            drop_determiners=False)
print('size of train surf 1grams bag:', len(train_surf_1grams_bag))

test_lemm_tacy_doc = textacy.Doc(test_lemm_tacy_doc)
test_surf_tacy_doc = textacy.Doc(test_surf_tacy_doc)

test_lemm_1grams_bag = test_lemm_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower', named_entities=False,
                                                          weighting='count', as_strings=True, filter_stops=False,
                                                          filter_punct=False, filter_nums=False, drop_determiners=False)
print('size of test lemm 1grams bag:', len(test_lemm_1grams_bag))

test_surf_1grams_bag = test_surf_tacy_doc.to_bag_of_terms(ngrams=1, normalize='lower', named_entities=False,
                                                          weighting='count', as_strings=True, filter_stops=False,
                                                          filter_punct=False, filter_nums=False, drop_determiners=False)
print('size of test surf 1grams bag:', len(test_surf_1grams_bag))

# %%
# test code
print(type(train_lemm_2grams_bag), len(train_lemm_2grams_bag))
print(type(train_lemm_1grams_bag), len(train_lemm_2grams_bag))
print('him . ', train_lemm_2grams_bag['him .'])
print('. the', train_lemm_2grams_bag['. the'])
i = 0
for sent in train_lemm_tacy_sents:
    print(sent.text)
    i += 1
    if i > 10: break

# test code
# for i,chs in enumerate(zip(train_lemm_tacy_doc.tokens,train_surf_tacy_doc.tokens)):
#     # if chs[0].text=='have' and chs[1].text=="'":
#     #     print(i,chs[0],chs[1])
#     #     break
#     if chs[0].text not in ['be','find','get','have','a','he','lie','use','leave','go','see','she','we','i','would'] and chs[0].text[0]!=chs[1].text[0]:
#         print(i,chs[0],chs[1])
#         break
#     # if i>=740 and i<=750:
#     #     print(i,chs[0],chs[1])
#
# # print(train_lemm_corpus[0:200])
# for i,chs in enumerate(zip(train_lemm_tacy_doc.tokens,train_lemm_corpus.split(' '))):
#     if chs[0].text!=chs[1]:
#         print(i,'|'+chs[0].text+'|','|'+chs[1]+'|')
#         # break
#     if i>345:
#         break

# %%
'''
Get all pair of surf-lemma and their count on train data set.
'''
pairs_list = []
for lemma, surf in zip(train_lemm_tacy_doc, train_surf_tacy_doc):
    pairs_list.append(surf.text.strip() + ' ' + lemma.text.strip())
train_surf_lemm_map = {}
for i, pair in enumerate(pairs_list):
    if pair not in train_surf_lemm_map:
        train_surf_lemm_map[pair] = pairs_list.count(pair)

# test code
print('are be ', train_surf_lemm_map['are be'])
# print('( ( ',train_surf_lemm_map['( ('])
# print('. . ',train_surf_lemm_map['. .'])

# %%
# test code
# print('(rimatara reed) ',train_lemm_2grams_bag['rimatara reed'])
print('(you be) ', train_lemm_2grams_bag['you be'])
print('(he go) ', train_lemm_2grams_bag['he go'])
print('p(be|you)=', train_lemm_2grams_bag['you be'] / train_lemm_1grams_bag['you'])
print('p(cat|a)=', train_lemm_2grams_bag['a cat'] / train_lemm_1grams_bag['a'])
print('p(am|i)=', train_surf_2grams_bag['i am'] / train_surf_1grams_bag['i'])
print('p(be-o|are-s)=', train_surf_lemm_map['are be'] / train_surf_1grams_bag['are'])
print('p(.-o|.-s)=', train_surf_lemm_map['. .'] / train_surf_1grams_bag['.'])
# print('p(the|bos)=',train_surf_2grams_bag['. the'])


# %%
'''
Functions of Evalutate the prediction
'''


def count_accuracy_raw(pred_corpus, target_corpus):
    """
    Test accuracy, Raw accuracy
    """
    count_accu = 0
    total = 0
    pred_sents = pred_corpus.split('.')
    target_sents = target_corpus.split('.')
    for pred_sent, target_sent in zip(pred_sents, target_sents):
        pred_list = pred_sent.split(' ')
        targ_list = target_sent.split(' ')
        for pred_token, target_token in zip(pred_list, targ_list):
            total += 1
            if pred_token == target_token:
                count_accu += 1
    return count_accu, total


raw_acc_count, raw_count_total = count_accuracy_raw(train_lemm_corpus, train_surf_corpus)
print('test of Accuracy raw:', raw_acc_count, '/', raw_count_total, '=', raw_acc_count / raw_count_total)


def count_accuracy_spacy_raw(pred_sents, target_sents):
    """
    Test accuracy,  accuracy of spacy's token
    """
    count_accu = 0
    total = 0
    for pred_sent, target_sent in zip(pred_sents, target_sents):
        total += 1
        for pred_token, target_token in zip(pred_sent, target_sent):
            total += 1
            if pred_token.text == target_token.text:
                count_accu += 1
    return count_accu, total


spacy_acc_count, spacy_count_total = count_accuracy_spacy_raw(train_lemm_tacy_sents, train_surf_tacy_sents)
print('test of Accuracy spacy:', spacy_acc_count, '/', spacy_count_total, '=', spacy_acc_count / spacy_count_total)


# this function is for when we want stop it before all sentences.
# if not, utilse metric.accuracy instead
def count_accuracy(pred_sents, target_sents):
    count_accu = 0
    total = 0
    for pred_sent, target_sent in zip(pred_sents, target_sents):
        pred_list = re.split(r"-| |\?", pred_sent)
        # pred_list=pred_sent.split(' ')
        for pred_token, target_token in zip(pred_list, target_sent):
            total += 1
            if pred_token == target_token.text:
                count_accu += 1
    return count_accu, total


def decode_sents(vectors, type_list):
    sents = []
    for v in vectors:
        sent = ' '.join(map(lambda x: type_list[x], v))
        # print (sent)
        sents.append(sent)
    return sents


def decode_sent(vector, type_list):
    return ' '.join(map(lambda x: type_list[x], vector))


# %%
'''
**** Model Bi-gramms predicteur ****
'''

'''
Get all  [lemm(t-1),lemm(t)] -> surf(t) 
and get map of bi-gramms [lemm(t-1),lemm(t)] -> surf word , 
in which the surface word is max count of the same pair of [lemm(t-1),lemm(t)].
for example: if there have {[you be]->are} 3 times, and {[you be]->is} 1 times,
then map([you be])=are.
'''
bigramms_lemm_surf_map = {}
bigramms_lemm_surf_count_map = {}
for lemm_sent, surf_sent in zip(train_lemm_tacy_sents, train_surf_tacy_sents):
    for i, token in enumerate(zip(lemm_sent, surf_sent)):
        if i == 0:
            if token[0].text in bigramms_lemm_surf_count_map:
                l1 = bigramms_lemm_surf_count_map[token[0].text]
                l1.append(token[1].text)
            else:
                bigramms_lemm_surf_count_map[token[0].text] = [token[1].text]
            lemm_pre = token[0].text
        else:
            if lemm_pre + ' ' + token[0].text in bigramms_lemm_surf_count_map:
                l1 = bigramms_lemm_surf_count_map[lemm_pre + ' ' + token[0].text]
                l1.append(token[1].text)
            else:
                bigramms_lemm_surf_count_map[lemm_pre + ' ' + token[0].text] = [token[1].text]
            lemm_pre = token[0].text

for k, v in bigramms_lemm_surf_count_map.items():
    word_counts = Counter(v)
    bigramms_lemm_surf_map[k] = word_counts.most_common(1)[0][0]

print('size of bi-grammes: ', len(bigramms_lemm_surf_map))

# test code
print('you be -> ', bigramms_lemm_surf_map['you be'])

# %%
'''
Model Bi-gramms predicteur predict on test data
'''
print('--Model Bi-gramms predicteur predict on test data:---')
bigramms_pred_sents = []
count_accu = 0
for k, sent in enumerate(zip(test_lemm_tacy_sents, test_surf_tacy_sents)):
    pred_sent = []
    for i, token in enumerate(zip(sent[0], sent[1])):
        if i == 0:
            if token[0].text in bigramms_lemm_surf_map:
                pred_token = bigramms_lemm_surf_map[token[0].text]
                if pred_token == token[1].text:
                    count_accu += 1
                pred_sent.append(pred_token)
            else:
                # if can't find the pair of this lemm word,use directly this lemm word
                pred_sent.append(token[0].text)

                # if this not paired lemm word ==the surface word correspondant.
                if token[0].text == token[1].text:
                    count_accu += 1
            lemm_pre = token[0].text
        else:
            if lemm_pre + ' ' + token[0].text in bigramms_lemm_surf_map:
                pred_token = bigramms_lemm_surf_map[lemm_pre + ' ' + token[0].text]
                if pred_token == token[1].text:
                    count_accu += 1
                pred_sent.append(pred_token)
            else:
                # if can't find the pair of this lemm word,use directly this lemm word
                pred_sent.append(token[0].text)

                # if this not paired lemm word ==the surface word correspondant.
                if token[0].text == token[1].text:
                    count_accu += 1
            lemm_pre = token[0].text

    pred_sent_text = ' '.join(pred_sent)
    # pred_sent_text=pred_sent_text.rstrip()
    bigramms_pred_sents.append(pred_sent_text)
    if k <= 30:
        print('-- NO.', k)
        print(test_lemm_tacy_sents[k].text)
        print(test_surf_tacy_sents[k].text)
        print(pred_sent_text)

# %%
'''
Calcule accuracy of Bi-gramme model:
'''
raw_acc_count, raw_count_total = count_accuracy_raw(test_lemm_corpus, test_surf_corpus)
print('Accuracy raw on test data:', raw_acc_count, '/', raw_count_total, '=', raw_acc_count / raw_count_total)

test_surf_tacy_sents_raw = [sent.text for sent in test_surf_tacy_sents]
from metric import *

taux_accu = accuracy(test_surf_tacy_sents_raw, bigramms_pred_sents)
print('Accuracy of bi-gramms predicteur on test data:', count_accu, '/', len(test_surf_tacy_doc), '=', taux_accu)

end_time = time.time()
print('The Bi-grammes took a total of %.3f minutes to do training and prediction.' % ((end_time - start_time) / 60))

# %%
'''
# Part-of-speech tagging
'''
# alternative for parse:nlp = spacy.load('en', disable=['parser', 'tagger']),tagger = Tagger(nlp.vocab)
nlp2 = spacy.load('en')

start_time = time.time()

parse_pred_sents = []
for i, sent in enumerate(bigramms_pred_sents):
    parsed_sent = nlp2(sent)
    parse_pred_sent = []
    rule1 = False
    rule2 = False
    rule3 = False
    rule4 = False
    rule42 = False
    rule43 = False
    for j, token in enumerate(parsed_sent):
        if token.dep_ == 'nsubj' and token.tag_ == 'NN':  # noun, singular or mass
            rule1 = True
        if token.dep_ == 'nsubj' and token.tag_ == 'NNS' or token.dep_ == 'expl':
            rule2 = True
        # this rule is not so good:
        # if token.pos_=='NUM':
        #     rule3=True
        if token.dep_ == 'pobj' and token.tag_ == 'CD' and len(token.text) == 4:  # 1990
            rule4 = True
        if rule4 and token.dep_ == 'nsubj' and token.tag_ == 'NN':
            rule42 = True
            rule4 = False
        if rule4 and (token.dep_ == 'nsubj' and token.tag_ == 'NNS' or token.dep_ == 'expl'):
            rule43 = True
            rule4 = False

        if rule1 and token.pos_ == 'VERB':
            rule1 = False
            if token.text == 'be':
                parse_pred_sent.append('is')
                continue
            if token.text == 'have':
                parse_pred_sent.append('has')
                continue
            if token.text == token.lemma_:
                parse_pred_sent.append(token.text + 's')
                continue

        if rule2 and token.pos_ == 'VERB':
            rule2 = False
            if token.text == 'be':
                parse_pred_sent.append('are')
                continue
            if token.text == 'has':
                parse_pred_sent.append('have')
                continue

        if rule3 and token.tag_ == 'NN':
            rule3 = False
            if token.text == token.lemma_:
                parse_pred_sent.append(token.text + 's')
                continue

        if rule42 and token.pos_ == 'VERB':
            rule42 = False
            if token.text in ['be', 'is']:
                parse_pred_sent.append('was')
                continue

            # this rule is not so good:
            # if token.text==token.lemma_ and token.text.endswith('e'):
            #     parse_pred_sent.append(token.text+'d')
            #     # print(' '.join(parse_pred_sent))
            #     continue

        if rule43 and token.pos_ == 'VERB':
            rule43 = False
            if token.text in ['be', 'are']:
                parse_pred_sent.append('were')
                continue
            # this rule is not so good:
            # if token.text==token.lemma_ and token.text.endswith('e'):
            #     parse_pred_sent.append(token.text+'d')
            #     # print(' '.join(parse_pred_sent))
            #     continue

        parse_pred_sent.append(token.text)
    parse_pred_sents.append(' '.join(parse_pred_sent))

taux_accu = accuracy(test_surf_tacy_sents_raw, parse_pred_sents)
print('Accuracy of Parse predicteur on test data:', taux_accu)

end_time = time.time()
print('The Parse took a total of %.3f minutes to do training and prediction.' % ((end_time - start_time) / 60))

# %%
# test code
# parse_pred_sent=[]
# parsed_sent=nlp2(bigramms_pred_sents[2371]) #772,123,2371
# rule1=False
# for j,token in enumerate( parsed_sent):
#     print(token.text, token.pos_, token.tag_, token.dep_)
#     if token.dep_=='nsubj' and token.tag_=='NN':
#         rule1=True
#     if rule1 and token.pos_=='VERB':
#         rule1=False
#         if token.text=='be':
#             parse_pred_sent.append('is')
#             continue
#         if token.text=='have':
#             parse_pred_sent.append('has')
#             continue
#         if token.text==token.lemma_:
#             parse_pred_sent.append(token.text+'s')
#             continue
#     parse_pred_sent.append(token.text)
# print(' '.join(parse_pred_sent))
