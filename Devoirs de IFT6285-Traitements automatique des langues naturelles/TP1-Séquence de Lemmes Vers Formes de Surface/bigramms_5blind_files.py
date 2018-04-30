#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# bigramms_5blind_files.py
# @author Zhibin.LU
# @created Fri Feb 23 2018 17:14:32 GMT-0500 (EST)
# @last-modified Sat Mar 31 2018 19:05:28 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# # # #

#%%
import gzip
import time
from collections import Counter

import regex as re
import spacy
import textacy
# import loader

import os
os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6285-Traitements automatique des langues naturelles/TP1/ift6285-tp1")
print(os.getcwd())

'''
Load text in a string.
'''

def loadData2str(corpuspath):
    with gzip.open(corpuspath, 'rt', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    input_words=[]
    target_words=[]
    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue
        line=line.encode("ascii", errors="ignore").decode()
        if len(line.split('\t'))==2:
            target_word, input_word = line.split('\t')
            input_word=input_word.lower().strip()
            target_word=target_word.lower().strip()
            pattern = re.compile(r'\'')
            input_word=re.sub(pattern, '', input_word)
            target_word=re.sub(pattern, '', target_word)

            input_word=re.sub("([\?\!\~\&\=\[\]\{\}\<\>\(\)\_\-\+\/\.])", r" \1 ",input_word)
            target_word=re.sub("([\?\!\~\&\=\[\]\{\}\<\>\(\)\_\-\+\/\.])", r" \1 ",target_word)

            #1990s
            pattern = re.compile(r'\d+s')
            m1=re.search(pattern, input_word)
            m2=re.search(pattern, target_word)
            if m2 is not None and m1 is None:
                input_word=re.sub('(\d+)', r"\1s", input_word)

            input_word=re.sub('(\d+)', r" \1 ", input_word)
            target_word=re.sub('(\d+)', r" \1 ", target_word)

            input_word=re.sub(' +', ' ', input_word)
            target_word=re.sub(' +', ' ', target_word)
            if input_word=='':
                continue
            input_words.append(input_word)
            target_words.append(target_word)
    return ' '.join(input_words),' '.join(target_words)

def loadData2str4blind(corpuspath):
    with open(corpuspath, 'rt', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    input_words=[]
    # target_words=[]
    for line in lines:
        if line.startswith('#begin') or line.startswith('#end'):
            continue
        line=line.encode("ascii", errors="ignore").decode()

        input_word = line
        input_word=input_word.lower().strip()
        pattern = re.compile(r'\'')
        input_word=re.sub(pattern, '', input_word)

        input_word=re.sub("([\?\!\~\&\=\[\]\{\}\<\>\(\)\_\-\+\/\.])", r" \1 ",input_word)

        #1990s
        input_word=re.sub('(\d+)', r"\1s", input_word)

        input_word=re.sub('(\d+)', r" \1 ", input_word)

        input_word=re.sub(' +', ' ', input_word)
        if input_word=='':
            continue
        input_words.append(input_word)
    return ' '.join(input_words)

train_lemm_corpus,train_surf_corpus=loadData2str('data/train-1183.gz')
blind1_lemm_corpus=loadData2str4blind('data/blind-995.txt')
blind2_lemm_corpus=loadData2str4blind('data/blind-996.txt')
blind3_lemm_corpus=loadData2str4blind('data/blind-997.txt')
blind4_lemm_corpus=loadData2str4blind('data/blind-998.txt')
blind5_lemm_corpus=loadData2str4blind('data/blind-999.txt')

train_lemm_corpus=re.sub(' +', ' ', train_lemm_corpus)
train_surf_corpus=re.sub(' +', ' ', train_surf_corpus)
blind1_lemm_corpus=re.sub(' +', ' ', blind1_lemm_corpus)
blind2_lemm_corpus=re.sub(' +', ' ', blind2_lemm_corpus)
blind3_lemm_corpus=re.sub(' +', ' ', blind3_lemm_corpus)
blind4_lemm_corpus=re.sub(' +', ' ', blind4_lemm_corpus)
blind5_lemm_corpus=re.sub(' +', ' ', blind5_lemm_corpus)

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
blind1_lemm_tacy_doc = nlp(blind1_lemm_corpus)
blind2_lemm_tacy_doc = nlp(blind2_lemm_corpus)
blind3_lemm_tacy_doc = nlp(blind3_lemm_corpus)
blind4_lemm_tacy_doc = nlp(blind4_lemm_corpus)
blind5_lemm_tacy_doc = nlp(blind5_lemm_corpus)

print('Tokens of train_lemm_tacy_doc: ', len(train_lemm_tacy_doc))
print('Tokens of train_surf_tacy_doc: ', len(train_surf_tacy_doc))
if len(train_lemm_tacy_doc) != len(train_surf_tacy_doc):
    print('Warning: the numbre of tokens of lemme and surfaceis in train not equal !!!!!!')

print('Tokens of blind1_lemm_tacy_doc: ', len(blind1_lemm_tacy_doc))
print('Tokens of blind2_lemm_tacy_doc: ', len(blind2_lemm_tacy_doc))
print('Tokens of blind3_lemm_tacy_doc: ', len(blind3_lemm_tacy_doc))
print('Tokens of blind4_lemm_tacy_doc: ', len(blind4_lemm_tacy_doc))
print('Tokens of blind5_lemm_tacy_doc: ', len(blind5_lemm_tacy_doc))

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

def getSentsFromTacyDoc(tacy_doc):
    sents = []
    start_ind = 0
    for token in tacy_doc:
        if token.text in ['.', '?', '!']:
            sents.append(tacy_doc[start_ind:token.i + 1])
            start_ind = token.i + 1
    return sents

blind1_lemm_tacy_sents=getSentsFromTacyDoc(blind1_lemm_tacy_doc)
blind2_lemm_tacy_sents=getSentsFromTacyDoc(blind2_lemm_tacy_doc)
blind3_lemm_tacy_sents=getSentsFromTacyDoc(blind3_lemm_tacy_doc)
blind4_lemm_tacy_sents=getSentsFromTacyDoc(blind4_lemm_tacy_doc)
blind5_lemm_tacy_sents=getSentsFromTacyDoc(blind5_lemm_tacy_doc)

print('total sentence of blind1,2,3,4 lemms:', len(blind1_lemm_tacy_sents),len(blind2_lemm_tacy_sents),len(blind3_lemm_tacy_sents),len(blind4_lemm_tacy_sents))

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


# %%
'''
Model Bi-gramms produit suface text on blind data
'''
print('--Model Bi-gramms suface text on blind data:---')

def predict2text(lemm_tacy_sents):
    bigramms_pred_sents = []
    bigramms_pred_sents_with_lemm = []
    count_accu = 0
    for k, sent in enumerate(lemm_tacy_sents):
        pred_sent = []
        pred_sent_with_lemm = []
        for i, token in enumerate(sent):
            if i == 0:
                if token.text in bigramms_lemm_surf_map:
                    pred_token = bigramms_lemm_surf_map[token.text]
                    pred_sent.append(pred_token)
                    pred_sent_with_lemm.append(pred_token.text+'\t'+token+'\n')
                else:
                    # if can't find the pair of this lemm word,use directly this lemm word
                    pred_sent.append(token.text)
                    pred_sent_with_lemm.append(token.text+'\t'+token.text+'\n')
                lemm_pre = token.text
            else:
                if lemm_pre + ' ' + token.text in bigramms_lemm_surf_map:
                    pred_token = bigramms_lemm_surf_map[lemm_pre + ' ' + token.text]
                    pred_sent.append(pred_token)
                    pred_sent_with_lemm.append(pred_token.text+'\t'+token+'\n')
                else:
                    # if can't find the pair of this lemm word,use directly this lemm word
                    pred_sent.append(token.text)
                    pred_sent_with_lemm.append(token.text+'\t'+token.text+'\n')
                    # if this not paired lemm word ==the surface word correspondant.
                lemm_pre = token.text

        pred_sent_text = ' '.join(pred_sent)
        pred_sent_text_with_lemme = ''.join(pred_sent_with_lemm)
        bigramms_pred_sents.append(pred_sent_text)
        bigramms_pred_sents_with_lemm.append(pred_sent_text_with_lemme)
        if k <= 5:
            print('-- NO.', k)
            print(pred_sent_text)
            print(pred_sent_text_with_lemme)
    return bigramms_pred_sents,bigramms_pred_sents_with_lemm

predi_blind1,predi_blind1_with_lemm=predict2text(blind1_lemm_tacy_sents)
predi_blind2,predi_blind2_with_lemm=predict2text(blind2_lemm_tacy_sents)
predi_blind3,predi_blind3_with_lemm=predict2text(blind3_lemm_tacy_sents)
predi_blind4,predi_blind4_with_lemm=predict2text(blind4_lemm_tacy_sents)
predi_blind5,predi_blind5_with_lemm=predict2text(blind5_lemm_tacy_sents)

# #%%
# with open('blind1_prediction.txt','w') as f:
#     for sent in predi_blind1_with_lemm:
#         f.write(sent)
#         f.write('\n')

# %%
'''
# Part-of-speech tagging
'''
# alternative for parse:nlp = spacy.load('en', disable=['parser', 'tagger']),tagger = Tagger(nlp.vocab)
nlp2 = spacy.load('en')

start_time = time.time()

def parse_modify(bigramms_pred_sents):
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

            # if token.dep_ == 'pobj' and token.tag_ == 'CD' and len(token.text) == 4:  # 1990
            #     rule4 = True
            # if rule4 and token.dep_ == 'nsubj' and token.tag_ == 'NN':
            #     rule42 = True
            #     rule4 = False
            # if rule4 and (token.dep_ == 'nsubj' and token.tag_ == 'NNS' or token.dep_ == 'expl'):
            #     rule43 = True
            #     rule4 = False

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

            # if token.text == 'be':
            #     parse_pred_sent.append('is')
            #     continue

            # if rule3 and token.tag_ == 'NN':
            #     rule3 = False
            #     if token.text == token.lemma_:
            #         parse_pred_sent.append(token.text + 's')
            #         continue

            # if rule42 and token.pos_ == 'VERB':
            #     rule42 = False
            #     if token.text in ['be', 'is']:
            #         parse_pred_sent.append('was')
            #         continue

            # this rule is not so good:
            # if token.text==token.lemma_ and token.text.endswith('e'):
            #     parse_pred_sent.append(token.text+'d')
            #     # print(' '.join(parse_pred_sent))
            #     continue

            # if rule43 and token.pos_ == 'VERB':
            #     rule43 = False
            #     if token.text in ['be', 'are']:
            #         parse_pred_sent.append('were')
            #         continue

            # this rule is not so good:
            # if token.text==token.lemma_ and token.text.endswith('e'):
            #     parse_pred_sent.append(token.text+'d')
            #     # print(' '.join(parse_pred_sent))
            #     continue

            parse_pred_sent.append(token.text)
        parse_pred_sents.append(parse_pred_sent)
    return parse_pred_sents

#%%
def generat_pred_file(filepath,predi_blind_sents,blind_lemm_tacy_sents):
    parse_pred_sents=parse_modify(predi_blind_sents)
    with open(filepath,'w') as f:
        for pred_sent,lemm_sent in zip(parse_pred_sents,blind_lemm_tacy_sents):
            if len(pred_sent)!=len(lemm_sent):
                print('Warning! length error!')
            for w1,w2 in zip(pred_sent,lemm_sent):
                f.write(w1+'\t'+w2.text)
                f.write('\n')
            f.write('\n')

generat_pred_file('blind1_bigrammes_and_parse.txt',predi_blind1,blind1_lemm_tacy_sents)
generat_pred_file('blind2_bigrammes_and_parse.txt',predi_blind2,blind2_lemm_tacy_sents)
generat_pred_file('blind3_bigrammes_and_parse.txt',predi_blind3,blind3_lemm_tacy_sents)
generat_pred_file('blind4_bigrammes_and_parse.txt',predi_blind4,blind4_lemm_tacy_sents)
generat_pred_file('blind5_bigrammes_and_parse.txt',predi_blind5,blind5_lemm_tacy_sents)