#Multioutput PyCoalescedTM

import pickle
import gzip
import sys
sys.path.append('../../PyCoalescedTsetlinMachineCUDA/')

from PyCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine

from sklearn.model_selection  import train_test_split
import argparse
import logging
import numpy as np

filename = 'simple_bag_of_words_features.pkl.gz'

with gzip.open(filename, 'rb') as file:
    loaded_features = pickle.load(file)


def make_label_vectors(labels_list, labels_dict):
    y_all = np.zeros((len(labels_list), len(labels_dict)), dtype=np.uint32)
    for lbl in range(len(labels_list)):
        for indv_label in labels_list[lbl]:
            y_all[lbl][indv_label] = 1
    return y_all
    


data = loaded_features['featurized'].astype("uint32") 
lb= loaded_features['labels']
labeldict = loaded_features['labels_:_labelnum']

y_all = make_label_vectors(lb, labeldict)

print(data.shape)
print(y_all.shape)

average_accuracy = 0.0


x_train, x_test, y_train, y_test = train_test_split(data, y_all)
x_train_ids=x_train[:,-1]
x_test_ids=x_test[:,-1]
x_train=x_train[:,:-1]
x_test=x_test[:,:-1]


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

for i in range(100):
	tm = MultiOutputTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

	tm.fit(x_train, y_train, epochs=200)

	print("Accuracy:", 100*(tm.predict(x_test) == y_test).mean())

	average_accuracy += 100*(tm.predict(x_test) == y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))