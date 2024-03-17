import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from transformers import AdamW, get_linear_schedule_with_warmup
from loaddata import load_data
from model import bertaModel
from train import train_model
from five_person.ann import ann
from torch.nn.parallel import DataParallel
import argparse
import io
import csv
import random
import pickle
import json
import vocab

dataset='meld'#dataset name
nums=7
DEVICE = torch.device("cuda:0")
DEVICE1 = torch.device("cuda:1")



def run(dataset,nums):
    berta_model = bertaModel(nums=nums)
    berta_model=berta_model.to(DEVICE)
    berta_model= DataParallel(berta_model,[DEVICE,DEVICE1])
    if dataset=='meld':
        train_filename=""
        dev_filename=""
        test_filename=""

    if dataset=="emorynlp":
        train_filename=""
        dev_filename=""
        test_filename=""
        
    if dataset=="iemocap":
        train_filename=""
        dev_filename=""
        test_filename=""


    past_utterance,train_person=ann(dataset,"train",train_filename)
    past_utterance,dev_person=ann(dataset,"dev",dev_filename)
    past_utterance,test_person=ann(dataset,"test",test_filename)
    

    train_dataloader=load_data(dataset,train_filename,train_person)
    dev_dataloader=load_data(dataset,dev_filename,dev_person)
    test_dataloader=load_data(dataset,test_filename,test_person)
    

    best=train_model(dataset,nums,berta_model,epochs,train_dataloader,dev_dataloader,test_dataloader)
    print(best)
    return best


run(dataset,nums)





