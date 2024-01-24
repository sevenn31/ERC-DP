import numpy as np
import pandas as pd
import csv
import pickle
import re
import time
from datetime import timedelta
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *
from five_person.dataset_processors import MyMapDataset
from five_person.other_person import other_person
import os
import sys
from pathlib import Path
from five_person.MLP_LM import get_inputs,testing
import os
import sys
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
import pickle
import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
from transformers import AutoModel,AutoTokenizer
from torch.nn.parallel import DataParallel
import json
import vocab

sys.path.insert(0, os.getcwd())
start = time.time()


DEVICE = torch.device("cuda:0")
DEVICE1 = torch.device("cuda:1")


embed="bert-base"
op_dir=''
token_length=512
mode="512_head"
embed_mode="cls"
batch_size = int(32)
inp_dir=""
lr=5e-4
epochs=10
layer=11
jobid=0
n_hl = 12
hidden_dim = 768



def ann(dataset,flag,filename):
    hidden_features = []
    

    df=pd.read_csv(filename)
    past_utterance,df=other_person(dataset,df)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    map_dataset = MyMapDataset(past_utterance,tokenizer, token_length, DEVICE, mode)
    data_loader = DataLoader(dataset=map_dataset,batch_size=batch_size,shuffle=False,)

    model=model.to(DEVICE)
    model = DataParallel(model,[DEVICE,DEVICE1])
    print( "\ngpu mem alloc: ", round(torch.cuda.memory_allocated() * 1e-9, 2), " GB")
    print("starting to extract LM embeddings...")
    
    for input_ids in data_loader:
        with torch.no_grad():
            tmp = []
            bert_output = model(input_ids)
            for ii in range(n_hl):
                if embed_mode == "cls":
                    tmp.append(bert_output.last_hidden_state[:, ii, :].cpu().numpy())
                elif embed_mode == "mean":
                    tmp.append((bert_output[2][ii + 1].cpu().numpy()).mean(axis=1))
            hidden_features.append(np.array(tmp))
  
    Path(op_dir).mkdir(parents=True, exist_ok=True)
    pkl_file_name = flag+ "-"+dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl"
    file = open(os.path.join(op_dir, pkl_file_name), "wb")
    pickle.dump(hidden_features, file)
    file.close()
    print("extracting embeddings for {} dataset: DONE!".format(dataset))
    print("MLP-------------------------------------------------------------------")
    print("{} : {} : {} : {} : {}".format(dataset, embed, layer, mode, embed_mode))
    n_classes = 2
    seed = jobid
    np.random.seed(seed)
    tf.random.set_seed(seed)
    inputs= get_inputs(inp_dir,flag, dataset, embed, embed_mode, mode, layer,n_hl)
    personality=testing(df,dataset,inputs)
    return past_utterance,personality



