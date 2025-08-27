#Feature extraction

#CountVectorizer, max n_grams = 2


import pandas as pd
from sklearn.model_selection  import train_test_split
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import time
import numpy as np
import pickle
import gzip

import ast

import torch
from transformers import AutoTokenizer, AutoModel


nortokenizer = AutoTokenizer.from_pretrained(
    "ltg/norbert4-large"
)

def tokenizer_wrapper(text):
    return nortokenizer.tokenize(text)

def create_labels(df):
    df['emneord'] = df['emneord'].apply(ast.literal_eval)
    
    all_emneord = pd.Series(sum(df['emneord'], []))
    unique_emneord = all_emneord.drop_duplicates().sort_values().reset_index(drop=True)
    labels = pd.Series(unique_emneord.index.values, index=unique_emneord.values).to_dict()

    return labels

print('Loading...')
df = pd.read_csv('../../data/stortinget_dataset.csv.gz')


print(df.shape)

print(df.columns)

column_to_use = 'sammendrag'

drop_set_for_na = ['emneord']+[column_to_use]

print('NA drops ',drop_set_for_na)


df = df.dropna(subset=drop_set_for_na)
df = df[df['emneord'].map(len) > 2]

print(df.shape)


emnedict = create_labels(df)

train, test = train_test_split(df, test_size=0.2)

print('Splits:', train.shape, test.shape)

labels=[]
sents_train = []
sents_test = []
for ind,line in train.iterrows():
    vedtak_text = line[column_to_use]
    #vedtak_text=vedtak_text.translate(str.maketrans('','',string.punctuation))
    sents_train.append(vedtak_text)
    #eval(line['emneord'])
    templabels =line['emneord']
    templabellist = []
    for tl in templabels:
        templabellist.append(emnedict[tl])
    labels.append(templabellist)

for ind,line in test.iterrows():
    vedtak_text = line[column_to_use]
    #vedtak_text=vedtak_text.translate(str.maketrans('','',string.punctuation))
    sents_test.append(vedtak_text)
    #eval(line['emneord'])
    templabels =line['emneord']
    templabellist = []
    for tl in templabels:
        templabellist.append(emnedict[tl])
    labels.append(templabellist)

print('Train:', len(sents_train), ' Test:', len(sents_test))


vectorizer = CountVectorizer(tokenizer=tokenizer_wrapper, lowercase=False, vocabulary = nortokenizer.get_vocab(), binary=True)

print('Vectorizing....')
#print(sents_train)
X_train = vectorizer.fit_transform(sents_train)
feature_names = vectorizer.get_feature_names_out()

X_test = vectorizer.transform(sents_test)


data = np.append(X_train,X_test)

print('Saving....')
features_dict = {'featurized':data, 'labels':labels, 'labels_:_labelnum':emnedict, 'train_test_split':len(train), 'source':column_to_use}



with gzip.open('../../processed_data/norbertcountvec_1gram_all_features_new.pkl.gz', 'wb') as file:
    pickle.dump(features_dict, file, protocol=pickle.HIGHEST_PROTOCOL)







