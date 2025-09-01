# v1

## AIMS:

 - Test Vanilla TM 
 - Test Coalesced TM 
     - without much pre-processing. 

- Try only with “Komiteens tilråding”-part of the document.

- Baseline :
    - ALL the Data
    - scikit-learn count vectorizer, Max n_grams = 2, binarized
    - s = 1
 
### Quick Start

- Download all the data
  ```python  v1/code/get_data/scrape_stortinget.py```

- Preprocess
  ```python  v1/code/preprocessing/***.py```

- Train/Test
  ```python  v1/code/coalesced/coalescedTM_***.py```

### Cloning this repo

- install git-lfs
- git lfs clone https://github.com/cair/TsetlinMachineSubjectTaggingPilot.git

-- this downloads a large saved dataset in v1/data, such that preprocess and train/test can be used without waiting for scraping

### Files:

- Preprocessing options:

  -- Simple Bag of words
  -- Countvectorizer
  -- Countvectorizer with NORBERT4 Tokenizer

See more at code/preprocessing.

- Training/Testing

  -- code/coalesced
  -- code/sparsecoalesced

  -- v1/code/vanillaTM.py

```
sample1 -> [label1, label2]

is converted to:

sample1 -> label1
sample1 -> label2

Failing with following error:

self.clause_bank[:, :, 0:self.number_of_state_bits_ta - 1] = np.uint32(~0)
OverflowError: Python integer -1 out of bounds for uint32

Possibly due to all labels not being represented in training data.
```


