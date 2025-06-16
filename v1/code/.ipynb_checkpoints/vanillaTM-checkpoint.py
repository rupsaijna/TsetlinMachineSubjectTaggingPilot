#VanillaTM

import pickle
import gzip
import sys
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from sklearn.model_selection  import train_test_split
import argparse
import logging
import numpy as np

filename = 'simple_bag_of_words_features.pkl.gz'

with gzip.open(filename, 'rb') as file:
    loaded_features = pickle.load(file)

def count_labels(y_list):
	label_counts={}
	for y in y_list:
		if y in label_counts.keys():
			label_counts[y]+= 1
		else:
			label_counts[y]= 1
	return label_counts

print(loaded_features.keys())

data = loaded_features['featurized'].astype("uint32") 
lb= loaded_features['labels']
labeldict = loaded_features['labels_:_labelnum']

new_data=[]
new_label=[]
data_index=0
for label_list in lb:
    for indiv_label in label_list:
        new_data.append(data[data_index])
        new_label.append(indiv_label)
    data_index+= 1

## deleteing classes with single sample ##
new_label = np.asarray(new_label)	
label_counts = count_labels(new_label)
   
print(len(new_data))
print(len(new_label))

for key,val in label_counts.items():
	if val == 1:
		lblname = ''
		for labelname,labelnumber in labeldict.items():
			if labelnumber == key:
				lblname = labelname
		print(key, np.nonzero(new_label==key), lblname)
		to_delete = np.nonzero(new_label==key)[0][0]
		del new_data[to_delete]
		new_label = np.delete(new_label,to_delete)

#verifying the deletes .....
print(len(new_data))
print(len(new_label))

label_counts = count_labels(new_label)
for key,val in label_counts.items():
	if val == 1:
		lblname = ''
		for labelname,labelnumber in labeldict.items():
			if labelnumber == key:
				lblname = labelname
		print(key, np.nonzero(new_label==key), lblname)

## deleteing classes with single sample ##

new_data= np.asarray(new_data)
new_label= np.asarray(new_label).astype("uint32")

print('Data:',new_data.shape, new_data.dtype)
print('Labels:',new_label.shape, new_label.dtype)


tm = TMClassifier(
        1000,
        800, 2.0,
        platform="GPU",
        weighted_clauses=True,
        clause_drop_p=0.75
    )



epochs = 20

x_train, x_test, y_train, y_test = train_test_split(new_data, new_label)
x_train_ids=x_train[:,-1]
x_test_ids=x_test[:,-1]
x_train=x_train[:,:-1]
x_test=x_test[:,:-1]

label_counts2 = count_labels(y_train)        
for key,val in label_counts2.items():
	if val == 1:
		print('in y_train',key)

label_counts2 = count_labels(y_test)        
for key,val in label_counts2.items():
	if val == 1:
		print('in y_test',key)

results_accuracy=[]


for r in range(epochs):
	print('Run:',r)
	tm.fit(x_train, y_train)
	result = 100 * (tm.predict(x_test) == y_test).mean()
	results_accuracy.append(result)
	print('Epoch %d Accuracy 0.3%f',r,result)
    
# Using return_class_sums:True
for sample_ind in len(x_test):
    print('\nPredict with return_class_sums:',tm.predict(x_test[sample_ind], return_class_sums=True))
    print('Truth', y_test[sample_ind])

# Using predict_compute_class_sums
for sample_ind in len(x_test):
    print('\nPredict with predict_compute_class_sums:',tm.predict_compute_class_sums(x_test,sample_ind, clip_class_sum=False))
    print('Truth', y_test[sample_ind])


    


