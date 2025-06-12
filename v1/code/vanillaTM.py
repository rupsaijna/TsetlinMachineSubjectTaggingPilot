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


print(loaded_features.keys())

data = loaded_features['featurized'].astype("uint32") 
lb= loaded_features['labels']

new_data=[]
new_label=[]
data_index=0
for label_list in lb:
    for indiv_label in label_list:
        new_data.append(data[data_index])
        new_label.append(indiv_label)
    data_index+= 1
        


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

results_accuracy=[]

print(len(list(set(y_train))))

print(loaded_features['labels_:_labelnum'])

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


    


