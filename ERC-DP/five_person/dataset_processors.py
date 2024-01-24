import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
import math
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *


prompt="The personality is [MASK].[SEP]"

class MyMapDataset(Dataset):
    def __init__(self, past_utterance,tokenizer, token_length, DEVICE, mode):
        input_ids= meld_embeddings(
                past_utterance, tokenizer, token_length, mode
        )
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        self.input_ids = input_ids
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def load_meld_df(past_utterance):
    text=[]
    for i in past_utterance:
        text.append(prompt+i)
    token_len=[0]*len(past_utterance)
    df1= pd.DataFrame({"text":text,"token_len":token_len})
    print(df1)
    return df1

def meld_embeddings(past_utterance,tokenizer, token_length, mode):
    input_ids = []

    df = load_meld_df(past_utterance)

    for ind in df.index:
        tokens = tokenizer.tokenize(df["text"][ind])
        df.at[ind, "token_len"] = len(tokens)

    for ii in range(len(df)):
        text = df["text"][ii]
        input_ids.append(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    pad_to_max_length=True,
                )
            )

    print("loaded all input_ids and targets from the data file!")
    #print(input_ids[0])
    return input_ids

