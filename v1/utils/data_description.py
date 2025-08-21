import pandas as pd


df = pd.read_csv('../data/stortinget_dataset.csv.gz')

print('No Sammendrag & Vedtak, Only Text :',df[df[['sammendrag', 'vedtak']].isna().all(axis=1) & df['text'].notna()].shape[0])
print('No Text & Vedtak, Only Sammendrag :',df[df[['text', 'vedtak']].isna().all(axis=1) & df['sammendrag'].notna()].shape[0])
print('No Text & Sammendrag, Only Vedtak :',df[df[['text', 'sammendrag']].isna().all(axis=1) & df['vedtak'].notna()].shape[0])
print('No Sammendrag, Vedtak, Text :',df[df[['sammendrag', 'vedtak', 'text']].isna().all(axis=1)].shape[0])
print('No Sammendrag, Only Text & Vedtak :',df[df[['text', 'vedtak']].notna().all(axis=1) & df['text'].isna()].shape[0])
print('No Vedtak, Only Text & Sammendrag :',df[df[['text', 'sammendrag']].notna().all(axis=1) & df['vedtak'].isna()].shape[0])
print('No Text, Only Vedtak & Sammendrag :',df[df[['vedtak', 'sammendrag']].notna().all(axis=1) & df['text'].isna()].shape[0])
print('Text, Vedtak & Sammendrag :',df[df[['text', 'vedtak', 'sammendrag']].notna().all(axis=1)].shape[0])
print('Only text :',df[df[['text']].notna().all(axis=1)].shape[0])


print('Total:', df.shape[0])


'''
No Sammendrag & Vedtak, Only Text : 1
No Text & Vedtak, Only Sammendrag : 0
No Text & Sammendrag, Only Vedtak : 0
No Sammendrag, Vedtak, Text : 3951
No Sammendrag, Only Text & Vedtak : 0
No Vedtak, Only Text & Sammendrag : 6157
No Text, Only Vedtak & Sammendrag : 0
Total: 17221

'''