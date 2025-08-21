import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Import model
tokenizer = AutoTokenizer.from_pretrained(
    "ltg/norbert4-large"
)

df = pd.read_csv('../data/stortinget_dataset.csv.gz')

print(df.columns)