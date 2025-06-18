# Multioutput PyCoalescedTM

import pickle
import gzip
import sys
sys.path.append('../../../PyCoalescedTsetlinMachineCUDA/')
from PyCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine
from sklearn.metrics import classification_report
from sklearn.model_selection  import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import argparse
import logging
import numpy as np
import os

filename = '../../processed_data/countvectorizer_2grams_10000_features.pkl.gz'
outputfile = '../../results/coalesced_countvecbow_2grams_10000.txt'
masteroutputfile = '../../results/master_results.csv'

num_clauses = 7000
T = 10000
s = 1.0
epochs = 10
train_epochs = 1

fo = open(outputfile, 'a+')
fo.write('\nUsing : PyCoalescedTsetlinMachineCUDA \nEncoding: CountVectorized BOW')
fo.write('\nTM parameters : Clauses:'+str(num_clauses)+' T:'+str(T)+' s:'+str(s)+' \n')

with gzip.open(filename, 'rb') as file:
    loaded_features = pickle.load(file)


def make_label_vectors(labels_list, labels_dict):
    y_all = np.zeros((len(labels_list), len(labels_dict)), dtype=np.uint32)
    for lbl in range(len(labels_list)):
        for indv_label in labels_list[lbl]:
            y_all[lbl][indv_label] = 1
    return y_all



data = loaded_features['featurized']
split_id = loaded_features['train_test_split']
lb= loaded_features['labels']
labeldict = loaded_features['labels_:_labelnum']
sorted_label_names = []
sorted_label_names = sorted(labeldict, key=labeldict.get)
y_all = make_label_vectors(lb, labeldict)
fo.write('\nNgrams:'+str(loaded_features['max_n_grams'])+' Features:'+str(loaded_features['max_features'])+' Source:'+loaded_features['source']+' \n')


print(data.shape)
print(y_all.shape)
if split_id == -1:
    fo.write('\nData Shape : '+str(data.shape)+' \nNumber of Labels: '+str(len(labeldict))+' \n')
    x_train, x_test, y_train, y_test = train_test_split(data, y_all)
else:
    x_train = data[0].toarray().astype(np.uint32)
    x_test = data[1].toarray().astype(np.uint32)
    y_train = y_all[:split_id, :]
    y_test = y_all[split_id:, :]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)



average_accuracy = 0.0


tm = MultiOutputTsetlinMachine(num_clauses, T, s, q=60, boost_true_positive_feedback=0)

for i in range(epochs):
	print('Epoch ', i, ' training ...' )
	print(x_train.shape)
	tm.fit(x_train, y_train, epochs=train_epochs)

	prediction = tm.predict(x_test)

	print("Accuracy:", 100*(prediction == y_test).mean())

	average_accuracy += 100*(prediction == y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))

	#print('\n Confusion Matrix ',multilabel_confusion_matrix(y_test, prediction, labels=sorted_label_names ))

	cr= classification_report(y_test, prediction, target_names = sorted_label_names, zero_division=np.nan )

	if i%10 == 0:
		fo.write('\nEpoch '+str(i)+': Acc:'+str(100*(prediction == y_test).mean()) +' ' )
		print('\n Classification Report:\n',cr)
print('\n Classification Report:\n',cr)
fo.write('\nEpoch '+str(epochs)+': Average Acc:'+str(average_accuracy/(epochs)) +' ' )
fo.write('\nClassification Report:\n{}'.format(cr))
fo.close()   


fo = open(masteroutputfile, 'a+')

#running_file_name, input_file_name, result_filename, Number_samples_total, Number_labels, preprocess, TMType, Clauses, T, s, train_epochs, run_epochs,Avg_accuracy,Notes
fo.write(os.path.basename(__file__)+','+filename+','+outputfile+',')
fo.write(str(x_train.shape[0])+','+str(len(labeldict)))
fo.write(',CountVectorized BOW,Coalesced,')
fo.write(str(num_clauses)+','+str(T)+','+str(s)+','+str(train_epochs)+','+str(epochs)+',')
fo.write(str(average_accuracy/epochs)+',')
fo.write(str(accuracy_score(y_test, prediction)*100)+',')
fo.write(str(precision_recall_fscore_support(y_test, prediction, average='weighted', zero_division=np.nan)[0]) +',')
fo.write(str(precision_recall_fscore_support(y_test, prediction, average='weighted', zero_division=np.nan)[1]) +',')
fo.write(str(precision_recall_fscore_support(y_test, prediction, average='weighted', zero_division=np.nan)[2]))
fo.write(',NA\n')
fo.close()