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
from dataset_processors import MyMapDataset
from transformers import AutoModel, AutoTokenizer
from other_person import other_person
import os
import sys
from pathlib import Path
from MLP_LM import get_inputs,testing
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
batch_size = 32
inp_dir=""
layer=11
seed=43
n_hl = 12
network = "MLP"


def ann(dataset,flag,filename):
    hidden_features = []
    df=pd.read_csv(filename)
    past_utterance,df=other_person(dataset,df)
    
    tokenizer = AutoTokenizer.from_pretrained("")
    model = AutoModel.from_pretrained("")
    map_dataset = MyMapDataset(past_utterance,tokenizer, token_length, DEVICE)
    data_loader = DataLoader(dataset=map_dataset,batch_size=batch_size,shuffle=False,)

    model=model.to(DEVICE)
    model = DataParallel(model,[DEVICE,DEVICE1]) 
    for input_ids in data_loader:
        with torch.no_grad():
            tmp = []
            bert_output = model(input_ids)
            for ii in range(n_hl):
                tmp.append(bert_output.last_hidden_state[:, ii, :].cpu().numpy())
            hidden_features.append(np.array(tmp))
  
    Path(op_dir).mkdir(parents=True, exist_ok=True)
    pkl_file_name = flag+ "-"+dataset + "-" + embed + ".pkl"
    file = open(os.path.join(op_dir, pkl_file_name), "wb")
    pickle.dump(hidden_features, file)
    file.close()
    print("extracting embeddings for {} dataset: DONE!".format(dataset))
    print("MLP-------------------------------------------------------------------")
    n_classes = 2
    np.random.seed(seed)
    tf.random.set_seed(seed)
    inputs= get_inputs(inp_dir,flag, dataset, embed,layer,n_hl)
    personality=testing(df,dataset,inputs)
    return past_utterance,personality


