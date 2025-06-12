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



print(loaded_features.keys())

data = loaded_features['featurized'].astype("uint32") 
lb= loaded_features['labels']

labels = np.array([np.array(xi).astype("object") for xi in lb]).astype("object") ## Cannot make np array with ragged edges!!

print(data.shape)

average_accuracy = 0.0

for i in range(100):
	tm = MultiOutputTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

	tm.fit(X_train, Y_train, epochs=200)

	print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

	average_accuracy += 100*(tm.predict(X_test) == Y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))