# v1

## AIMS:

 - Test Vanilla TM 
 - Test Coalesced TM 
     - without much pre-processing. 

- Try only with “Komiteens tilråding”-part of the document.

### Files:

- v1/code/simple_bag_of_words.py

```Creates present/absent features w.r.t. bag of words. Saves following in simple_bag_of_words_features.pkl.gz

{'featurized':converted data, 
'labels':labels as number,
'idx_:_word': unique words with indices,
'word_:_idx':reverse word map, 
'labels_:_labelnum':dict mapping of labels to numbers}```

- v1/code/vanillaTM.py

```
sample1 -> [label1, label2]

is converted to:

sample1 -> label1
sample1 -> label2

Failing with following error:

self.clause_bank[:, :, 0:self.number_of_state_bits_ta - 1] = np.uint32(~0)
OverflowError: Python integer -1 out of bounds for uint32

Possibly due to all labels not being represented in training data.```

- v1/code/coalescedTM.py

```
Y needs to be np.array.
Labels are currently in a ragged-edge list. Cannot convert.

```


