#Feature extraction

#Bag of Words

import pandas as pd
#from sklearn.cross_validation import train_test_split
import numpy as np
#from nltk.util import ngrams,everygrams
import re
import string
import time
import numpy as np

import pickle
import gzip

import ast

def create_labels(df):
    df['emneord'] = df['emneord'].apply(ast.literal_eval)
    
    all_emneord = pd.Series(sum(df['emneord'], []))
    unique_emneord = all_emneord.drop_duplicates().sort_values().reset_index(drop=True)
    labels = pd.Series(unique_emneord.index.values, index=unique_emneord.values).to_dict()

    return labels


df = pd.read_csv('data2.csv')

print(df.shape)

df = df.dropna(subset=['tilrading', 'emneord'])
df = df[df['emneord'].map(len) > 2]

print(df.shape)


emnedict = create_labels(df)

sents=[]
labels=[]
all_words=['<UNK>', '<START>','<STOP>']

def encode_sentences(txt, word_set):
	feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
	tnum=0
	for t in txt:
		s_words=['<START>']+t[1:]+['<STOP>']
		for w in s_words:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		feature_set[tnum][-1]=t[0]  ##Set the last index to the sent 
		tnum+=1
	return feature_set

maxlen=0
lcnt=0

for ind,line in df.iterrows():
    vedtak_text = line['tilrading']
    vedtak_text=vedtak_text.translate(str.maketrans('','',string.punctuation))
    words=vedtak_text.split(' ')
    bl=list(set(words))
    all_words+=bl
    words.insert(0,ind)
    sents.append(words)
    #eval(line['emneord'])
    templabels =line['emneord']
    templabellist = []
    for tl in templabels:
        templabellist.append(emnedict[tl])
    labels.append(templabellist)

word_set=set(all_words)

i=0
word_idx = dict((c, i + 1) for i, c in enumerate(word_set,start = -1))
reverse_word_map = dict(map(reversed, word_idx.items()))

data=encode_sentences(sents, word_set)


list_len = [len(i) for i in labels]
print(max(list_len))

features_dict = {'featurized':data, 'labels':labels, 'idx_:_word':word_idx, 'word_:_idx':reverse_word_map, 'labels_:_labelnum':emnedict, 'train_test_split':-1}



with gzip.open('simple_bag_of_words_features.pkl.gz', 'wb') as file:
    pickle.dump(features_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

