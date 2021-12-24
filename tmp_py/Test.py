import pandas

import SMBF, MMBF
import pandas as pd
import random
import numpy as np
from Hash_All_FHI import SimpleFHI as BF

df = pd.read_csv("../file/file.csv")
col_size = 11


# method1:不拼接多个键值，而是采用多个BLoom Filter
def MMBF_FPR(positive, negative, filter_size, k):
    MMBF_ = MMBF.MMBF(positive, filtersize=filter_size, k=k)
    c = 0
    for n in negative:
        if MMBF_.query(n):
            c += 1
    return c / len(negative)


def SMBF_FPR(positive, negative, filter_size, k):
    SMBF_ = SMBF.MultiKeyBloomFilterWithSameHash(positive, filtersize=filter_size, k=k)
    c = 0
    for n in negative:
        if SMBF_.query(n):
            c += 1
    return c / len(negative)


def ISMBF_FPR(positive, negative, filter_size, k, interval, sample_num):
    I = []
    insert_list = np.array(positive)
    for i in range(len(insert_list[0])):
        s = random.sample(list(insert_list[:, i]), sample_num)
        temp = len(set(s)) / sample_num * interval // 1 + 1  # P*I//1+1
        if interval % 2 == 0:
            if temp > interval / 2:
                I.append(interval / 2 - temp)
            else:
                I.append(interval / 2 - temp + 1)
        else:
            if temp >= (interval + 1) / 2:
                I.append((interval + 1) / 2 - temp)
            else:
                I.append((interval + 1) / 2 - temp)
    IMLBF = SMBF.IntervalMultiKeyBloomFilterWithDifferentHash(insert_list, I, filtersize=filter_size, k=k)
    c = 0
    for n in negative:
        if IMLBF.query(n):
            c += 1
    return c / len(negative)


def MLBF_FPR(data, label, score, tau, filter_size, k):
    insert_list = []
    for i in range(len(data)):
        if label[i] == 1 and score[i] < tau:
            insert_list.append(data[i])
    MLBF = SMBF.MultiKeyBloomFilterWithSameHash(insert_list, filtersize=filter_size, k=k)
    neg_count = 0
    f = 0
    for i in range(len(data)):
        if label[i] == 0:
            neg_count += 1
        if label[i] == 0 and score[i] < tau:
            if MLBF.query(data[i]):
                f += 1
        if label[i] == 0 and score[i] >= tau:
            f += 1
        # if label[i]==1 and
    return f / neg_count


def IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num):
    insert_list = []
    for i in range(len(data)):
        if label[i] == 1 and score[i] < tau:
            insert_list.append(data[i])
    I = []
    insert_list = np.array(insert_list)
    for i in range(len(insert_list[0])):
        s = random.sample(list(insert_list[:, i]), sample_num)
        temp = len(set(s)) / sample_num * interval // 1 + 1  # P*I//1+1
        if interval % 2 == 0:
            if temp > interval / 2:
                I.append(interval / 2 - temp)
            else:
                I.append(interval / 2 - temp + 1)
        else:
            if temp >= (interval + 1) / 2:
                I.append((interval + 1) / 2 - temp)
            else:
                I.append((interval + 1) / 2 - temp)
    IMLBF = SMBF.IntervalMultiKeyBloomFilterWithDifferentHash(insert_list, I, filtersize=filter_size, k=k)
    neg_count = 0
    f = 0
    for i in range(len(data)):
        if label[i] == 0:
            neg_count += 1
        if label[i] == 0 and score[i] < tau:
            if IMLBF.query(data[i]):
                f += 1
        if label[i] == 0 and score[i] >= tau:
            f += 1
    return f / neg_count


def concat_BF(positive, negative, filter_size, k):
    BF_ = BF(positive, filtersize=filter_size, k=k)
    c = 0
    for n in negative:
        if BF_.query(n):
            c += 1
    return c / len(negative)

def CTR_Reader(size=59999):
    f = open("train.txt", "r")  # 设置文件对象
    lable = []
    data = []
    for i in range(size):
        line = f.readline().strip("\n").split("\t")
        lable.append(line[0])
        data.append(line[1:])
    return data,lable

if __name__ == '__main__':
    tau = 1
    interval = 4
    filter_size = 250000
    k = 7
    sample_num = 1000
    df = pandas.read_csv("../file/file_score.csv")
    label = df["label"]
    score = df["score"]
    data = []
    positive = []
    negative = []
    positive_concat = []
    negative_concat = []
    for i in range(df.shape[0]):
        if label[i] == 0:
            positive.append(list(df.iloc[i, :-2]))
            positive_concat.append("".join([str(x) for x in df.iloc[i, :-2]]))
        else:
            negative.append(list(df.iloc[i, :-2]))
            negative_concat.append("".join([str(x) for x in df.iloc[i, :-2]]))

    print("正例占比: ", len(positive), "负例占比: ", len(negative))
    print("Concat_BF,FPR: ", concat_BF(positive_concat, negative_concat, filter_size, k))
    print("MMBF,FPR: ", MMBF_FPR(positive, negative, filter_size, k))
    print("SMBF,FPR: ", SMBF_FPR(positive, negative, filter_size, k))
    for i in range(df.shape[0]):
        data.append(list(df.iloc[i, :-3]))
    data = np.array(data)
    for interval in range(3, 7):
        print("*****************")
        print("interval", interval)
        print("ISMBF,FPR: ", ISMBF_FPR(positive, negative, filter_size, k, interval, sample_num))
        print("MLBF,FPR: ", MLBF_FPR(data, label, score, tau, filter_size, k))
        print("IMLBF,FPR: ", IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num))
        print("*****************")
