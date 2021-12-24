import pandas

import SMBF, MMBF
import pandas as pd
import random
import numpy as np
from Hash_All_FHI import SimpleFHI as BF
import time
import os

df = pd.read_csv("file/file.csv")
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
    time1 = time.time()
    for n in negative:
        if SMBF_.query(n):
            c += 1
    time2 = time.time()
    print("SMBF CPU time for every element", (time2 - time1) / len(negative))
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
    time1 = time.time()
    for n in negative:
        if IMLBF.query(n):
            c += 1
    time2 = time.time()
    print("ISMBF CPU time for every element", (time2 - time1) / len(negative))
    return c / len(negative)


def MLBF_FPR(data, label, score, tau, filter_size, k):
    insert_list = []
    a = 0
    b = 0
    for i in range(len(data)):
        if label[i] == 1 and score[i] < tau:
            insert_list.append(data[i])
    MLBF = SMBF.MultiKeyBloomFilterWithSameHash(insert_list, filtersize=filter_size, k=k)
    neg_count = 0
    f = 0
    time1 = time.time()
    for i in range(len(data)):
        if label[i] == 0:
            neg_count += 1
        if label[i] == 0 and score[i] < tau:
            if MLBF.query(data[i]):
                f += 1
                a += 1
        if label[i] == 0 and score[i] >= tau:
            f += 1
            b += 1
        # if label[i]==1 and
    # print("BF错误个数：", a, "，分类器错误个数：", b)
    time2 = time.time()
    print("MLBF CPU time for every element", (time2 - time1) / neg_count)
    return f / neg_count


def IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num):
    insert_list = []
    for i in range(len(data)):
        if label[i] == 1 and score[i] <= tau:
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
    time1 = time.time()
    for i in range(len(data)):
        if label[i] == 0:
            neg_count += 1
        if label[i] == 0 and score[i] < tau:
            if IMLBF.query(data[i]):
                f += 1
        if label[i] == 0 and score[i] >= tau:
            f += 1
    time2 = time.time()
    print("IMLBF CPU time for every element", (time2 - time1) / neg_count)
    return f / neg_count


def CMLBF_FPR(data, label, score, tau, filter_size, k):
    tag = [random.sample('zyxwvutsrqponmlkjihgfedcba', 5) for _ in range(len(data[0]))]
    insert_list = []
    a = 0
    b = 0
    for i in range(len(data)):
        if label[i] == 1 and score[i] < tau:
            list1 = [str(data[i][k]) + str(tag[k]) for k in range(len(data[0]))]
            insert_list.append("".join(list1))
    BF_ = BF(insert_list, filtersize=filter_size, k=k)
    neg_count = 0
    f = 0
    time1 = time.time()
    for i in range(len(data)):
        if label[i] == 0:
            neg_count += 1
        if label[i] == 0 and score[i] < tau:
            list1 = [str(data[i][k]) + str(tag[k]) for k in range(len(data[0]))]
            if BF_.query("".join(list1)):
                f += 1
                a += 1
        if label[i] == 0 and score[i] >= tau:
            f += 1
            b += 1
        # if label[i]==1 and
    # print("BF错误个数：", a, "，分类器错误个数：", b)
    time2 = time.time()
    print("CMLBF CPU time for every element", (time2 - time1) / neg_count)
    return f / neg_count


def concat_BF(positive, negative, filter_size, k):
    BF_ = BF(positive, filtersize=filter_size, k=k)
    c = 0
    time1 = time.time()
    for n in negative:
        if BF_.query(n):
            c += 1
    time2 = time.time()
    print("concat-BF CPU time for every element", (time2 - time1) / len(negative))
    return c / len(negative)


def CTR_Reader(size=59999):
    f = open("train.txt", "r")  # 设置文件对象
    lable = []
    data = []
    tag = []
    for j in range(40):
        tag.append("".join(random.sample('zyxwvutsrqponmlkjihgfedcba', 7)))
    for i in range(size):
        line = f.readline().strip("\n").split("\t")
        lable.append(line[0])
        for j in range(1, len(line)):
            line[j] = line[j] + tag[j]
        data.append(line[1:])
    print(data[0])
    return data, lable


if __name__ == '__main__':
    ans = 1
    tau = 0.99
    interval = 6
    # filter_size = 22 * 1024 * 8
    k = 6
    sample_num = 1000
    df = pandas.read_csv("file/file_score.csv")
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
    # for k in range(20):
    #     print("k", k)
    #     print("Concat_BF,FPR: ", concat_BF(positive_concat, negative_concat, filter_size, k))
    # print("MMBF,FPR: ", MMBF_FPR(positive, negative, filter_size, k))

    for i in range(df.shape[0]):
        data.append(list(df.iloc[i, :-3]))
    data = np.array(data)
    best_tau = 1
    best_I = 1
    best_interval = 6
    min_FPR = 1
    filter_size = 120 * 1024
    for size in range(150, 240+1, 15):
        print("bitmap size:", size)
        filter_size = size * 1024
        for interval in range(2, 8+1, 2):
            fpr1 = ISMBF_FPR(positive, negative, filter_size, k, interval, sample_num)
            fpr2 = SMBF_FPR(positive, negative, filter_size, k)
            print("interval:", interval, ",优化FPR:", fpr1, "原始：", fpr2)
        print("*****************************")
    # for interval in range(1, 15):
    #     fpr = IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num)
    #     if fpr < min_FPR:
    #         best_interval = interval
    #     print("IMLBF,FPR: ", fpr, ",interval:", interval)
    # print("最优interval：", best_interval)
    interval = 6
    for size in range(150, 240+1, 15):
        filter_size = size * 1024
        print("bitmap大小：", size, "kb")
        # for interval in range(1, 15):
        #     temp = IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num)
        #     if temp < ans:
        #         ans = temp
        #         best_tau = tau
        #         best_I = interval
        #     for interval in range(1, 15):
        #         fpr = IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num)
        #         if fpr < min_FPR:
        #             best_interval = interval
        #             min_FPR = fpr
        # print("最优interval：", best_interval)
        print("SMBF,FPR: ", SMBF_FPR(positive, negative, filter_size, k))
        print("MLBF,FPR: ", MLBF_FPR(data, label, score, tau, filter_size, k))
        print("IMLBF,FPR: ", IMLBF_FPR(data, label, score, tau, interval, filter_size, k, sample_num))
        print("CMLBF,FPR: ", CMLBF_FPR(data, label, score, tau, filter_size, k))
        print("*********************")
    print(ans, best_tau, best_I)
