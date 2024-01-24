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
import torch

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

sys.path.insert(0, os.getcwd())


def get_inputs(inp_dir,flag, dataset, embed, embed_mode, mode, layer,n_hl):
    file = open(
        inp_dir +flag+ "-"+dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl", "rb"
    )
    data = pickle.load(file)
    data_x= list(data)
    file.close()

    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)
    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1
    inputs = []
    n_batches = len(data_x)
    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))

    inputs = np.array(inputs)
    return inputs


def testing(df,dataset,inputs):

    trait_labels ={}
    trait_labels["EXT"]=[]
    trait_labels["NEU"]=[]
    trait_labels["AGR"]=[]
    trait_labels["CON"]=[]
    trait_labels["OPN"]=[]
    print(df)
    for i in trait_labels.keys():
        model = keras.models.load_model("")#personality model  
        predictions = model.predict(inputs).tolist()
        x=[]
        for item in predictions:
            if item[0]>item[1]:
                x.append(0)
            else:
                x.append(1)
        trait_labels[i]=x
        num=0
        num1=0
        for j in x:
            if j ==0:
                num+=1
            else:
                num1+=1
        print(num,num1)

    five=[]
    for i in range(len(inputs)):
        x=[]
        x.append(trait_labels["EXT"][i])
        x.append(trait_labels["NEU"][i])
        x.append(trait_labels["AGR"][i])
        x.append(trait_labels["CON"][i])
        x.append(trait_labels["OPN"][i])
        five.append(x)
    people=df["people"].tolist()
    person=[]
    num=0
    for i in people:
        person.append(five[num])
        num=num+1
    return person




     