import numpy as np
import pandas as pd
import csv



def other_person(dataset,df):
    people=[]
    if dataset=='meld':
        person_dict={}
        Utterance=df["Utterance"].tolist()
        Speaker=df["Speaker"].tolist()
        Dialogue_ID=df["Dialogue_ID"].tolist()
    if dataset=='iemocap':
        person_dict={}
        Utterance=df["utterance"].tolist()
        Speaker=df["speaker"].tolist()
        Dialogue_ID=df["dialog"].tolist()
    if dataset=='emorynlp':
        person_dict={}
        Utterance,Speaker,Dialogue_ID=em_cal(df)
    for i in range(len(Utterance)):
        if i==0 or Dialogue_ID[i]!=Dialogue_ID[i-1]:#等于0或对话开始的时候
            if Speaker[i] in person_dict.keys():
                people.append(person_dict[Speaker[i]])    
            else:
               people.append(Utterance[i])
        if Dialogue_ID[i]==Dialogue_ID[i-1]:
            past_sentence=""
            for j in range(i,-1,-1):
                if Speaker[i]==Speaker[j] and Dialogue_ID[i]==Dialogue_ID[j]:
                    past_sentence=Utterance[j]+past_sentence
            people.append(past_sentence)
    df["people"]=people
    past_utterance=[]
    for i in people:
        if type(i)==str:
            past_utterance.append(i)
    print(len(past_utterance))
    print(df)
    return past_utterance,df


def em_cal(df):
    Utterance=df["Utterance"].tolist()
    Speaker=df["Speaker"].tolist()
    dialog=[]
    num=0
    x=df["Scene_ID"].tolist()
    for i in range(len(x)):
        if i==0:
            dialog.append(num)
        else:
            if x[i]==x[i-1]:
                dialog.append(num)
            if x[i]!=x[i-1]:
                num+=1
                dialog.append(num)
    return Utterance,Speaker,dialog



    
