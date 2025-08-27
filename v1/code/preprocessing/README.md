- v1/code/preprocessing/simple_bag_of_words.py

```Creates present/absent features w.r.t. bag of words. Saves following in processed_data/simple_bag_of_words_features.pkl.gz

{'featurized':converted data, 
'labels':labels as number,
'idx_:_word': unique words with indices,
'word_:_idx':reverse word map, 
'labels_:_labelnum':dict mapping of labels to numbers
'train_test_split':not split into train-test}
```

- v1/code/preprocessing/countvectorizer_bag_of_words.py

```Creates features w.r.t. CountVectorizer. Saves following in processed_data/countvectorizer_bag_of_words_features.pkl.gz

{'featurized':converted data, 
'labels':labels as number,
'idx_:_word': unique words with indices,
'word_:_idx':reverse word map,
'featurenames_vectorizer': feature names after calling CountVectorizer fit_transform,
'labels_:_labelnum':dict mapping of labels to numbers
'train_test_split':index of split for train-test}
```

- v1/code/preprocessing/norbert_countvec.py

  ```Creates features w.r.t. CountVectorizer using NORBERT4 tokenizer. Saves following in processed_data/norbertcountvec_1gram_all_features_new.pkl.gz

  Uses 1gram, and all NORBERT tokens(512000)

{'featurized':converted data, 
'labels':labels as number,
'labels_:_labelnum':dict mapping of labels to numbers
'train_test_split':index of split for train-test}
```
