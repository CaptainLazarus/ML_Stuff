import pandas as pd
import numpy as np

def inp():
    #Reading a DataFrame
    try:
        data_frame = pd.read_csv("test.csv" , sep="," , names=["Feature 1" , "Feature 2" , "Feature 3" , "Feature 4" , "Class"] , header=None)
    except FileNotFoundError as e:
        print(e)
        exit()
    if data_frame.empty:
        print("No data")
        exit()
    return data_frame

def classify(T,b,p_f):
    #Classification
    p = []
    k=0
    multMax = 0
    for j in b:
        m=1
        for feature in range(0,len(T)):
            value = T[feature]
            temp = p_f[feature][value][j]
            # print(temp)
            m*=temp
        print("p({}) -> {}".format(j , m))
        if m > multMax:
            multMax = m
            cl=j
    return cl

def create_Model(df): 
    train_data = df.iloc[:, :4]
    labels = df.iloc[:, 4:]

    #Class probabilities
    dic = dict.fromkeys(labels.values[:,0] , 0)
    for i in labels.values[:,0]:
        dic[i]+=1
    p_c = [dic[i]/labels.shape[0] for i in dic]


    #Feature probabilities
    dic2 = {}
    for feature in range(0,train_data.shape[1]):
        dic2[feature] = dict()
        for value in train_data.values[: , feature]:
            dic2[feature][value] = dict.fromkeys(labels.values[:,0] , 0)

    #Utility step
    for feature in range(0,train_data.shape[1]):
        index=0
        for value in train_data.values[: , feature]:
            temp = labels.values[:,0][index]
            dic2[feature][value][temp]+=1
            index+=1

    #Finding probability of P(x/"Class")
    for i in range(len(dic2)):
        vals = dic2[i]
        for j in vals:
            for k in dic.keys():
                dic2[i][j][k] = dic2[i][j][k]/dic[k]

    return dic,dic2

df = inp()
dic,dic2 = create_Model(df)

Input = [0 ,1,1,1]
print(classify(Input, dic , dic2))