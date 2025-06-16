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

max_ngram = 2
num_features = 10000


def create_labels(df):
    df['emneord'] = df['emneord'].apply(ast.literal_eval)
    
    all_emneord = pd.Series(sum(df['emneord'], []))
    unique_emneord = all_emneord.drop_duplicates().sort_values().reset_index(drop=True)
    labels = pd.Series(unique_emneord.index.values, index=unique_emneord.values).to_dict()

    return labels

def encode_sentences(txt, word_set):
	feature_set=np.zeros((len(txt), len(word_set)),dtype=int)
	tnum=0
	for t in txt:
		s_words=['<START>']+t[1:]
		for w in s_words:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		#feature_set[tnum][-1]=t[0]  ##Set the last index to the sent 
		tnum+=1
	return feature_set



df = pd.read_csv('../../data/data2.csv')

print(df.shape)

df = df.dropna(subset=['tilrading', 'emneord'])
df = df[df['emneord'].map(len) > 2]

print(df.shape)


emnedict = create_labels(df)

train, test = train_test_split(df, test_size=0.2)

labels=[]
all_words= []
sents_train = []
sents_test = []
for ind,line in train.iterrows():
    vedtak_text = line['tilrading']
    vedtak_text=vedtak_text.translate(str.maketrans('','',string.punctuation))
    words=vedtak_text.split(' ')
    bl=list(set(words))
    all_words+=bl
    sents_train.append(words)
    #eval(line['emneord'])
    templabels =line['emneord']
    templabellist = []
    for tl in templabels:
        templabellist.append(emnedict[tl])
    labels.append(templabellist)

for ind,line in test.iterrows():
    vedtak_text = line['tilrading']
    vedtak_text=vedtak_text.translate(str.maketrans('','',string.punctuation))
    words=vedtak_text.split(' ')
    bl=list(set(words))
    all_words+=bl
    sents_test.append(words)
    #eval(line['emneord'])
    templabels =line['emneord']
    templabellist = []
    for tl in templabels:
        templabellist.append(emnedict[tl])
    labels.append(templabellist)

word_set=set(all_words)

i=2
word_idx = dict((c, i + 1) for i, c in enumerate(word_set,start = -1))
word_idx["<PAD>"] = 0
word_idx["<START>"] = 1
word_idx["<UNK>"] = 2

reverse_word_map = dict(map(reversed, word_idx.items()))

vectorizer_X = CountVectorizer(
    tokenizer=lambda s: s,
    token_pattern=None,
    ngram_range=(1, max_ngram),
    lowercase=False,
    binary=True,
    max_features=num_features
)

#print(sents_train)
X_train = vectorizer_X.fit_transform(sents_train)
feature_names = vectorizer_X.get_feature_names_out()

X_test = vectorizer_X.transform(sents_test)




data = np.append(X_train,X_test)


features_dict = {'featurized':data, 'labels':labels, 'idx_:_word':word_idx, 'word_:_idx':reverse_word_map, 'featurenames_vectorizer': feature_names, 'labels_:_labelnum':emnedict, 'train_test_split':len(train)}



with gzip.open('../../processed_data/countvectorizer_bag_of_words_features.pkl.gz', 'wb') as file:
    pickle.dump(features_dict, file, protocol=pickle.HIGHEST_PROTOCOL)







