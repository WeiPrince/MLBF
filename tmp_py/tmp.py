import numpy as np
import pandas as pd
import SMBF
import random
f = open("train.txt", "r")  # 设置文件对象
positive_size = 100000
FP_size = 1200
negative_size = 100000-FP_size
positive_data = []
negative_data = []
while True:
    line = f.readline().strip("\n").split("\t")
    l = int(line[0])
    if l == 1:
        positive_data.append(line)
        positive_size -= 1
        if positive_size == 0:
            print("正例数据已经生成完成")
            break
positive_data = np.array(positive_data)
while True:
    line = f.readline().strip("\n").split("\t")
    l = int(line[0])
    if l == 0:
        neg = line
        exist = True
        count = 0
        for i in range(1, len(neg)):
            # print(list(positive_data[:, i]))
            if str(neg[i]) not in list(positive_data[:, i]):
                exist = False
                break
        if not exist:
            if len(negative_data) % 1000 == 0:
                print("找到", len(negative_data), "条负例数据")
            negative_data.append(line)
            negative_size -= 1
    if negative_size == 0 and FP_size==0:
        print("负例数据已经生成完成, 其中FP_size:", FP_size)
        break
data1 = np.vstack((positive_data, np.array(negative_data)))
print(len(data1))
datafram = pd.DataFrame(data1)
datafram.to_csv("CTR_train.csv", index=False)
print("生成完成")

#
#
# label = []
# data = []
# tag = []
# for j in range(40):
#     tag.append("".join(random.sample('zyxwvutsrqponmlkjihgfedcba', 15)))
# for i in range(100000):
#     line = []
#     label.append(int(data1.iloc[i, 0]))
#     for j in range(1, len(data1.iloc[i])):
#         line.append(str(data1.iloc[i, j]) + tag[j])
#     data.append(line)
# positive = []
# negative = []
# for i in range(len(data)):
#     if label[i] == 0:
#         positive.append(data[i])
#     else:
#         negative.append(data[i])
# SMBF_ = SMBF.MultiKeyBloomFilterWithSameHash(, filtersize=1000000000, k=7)
# df = pd.read_csv("score_CTR.csv")
#     # label = df["label"]
#     score = df["score"]
#     positive = []
#     negative = []
#     positive_concat = []
#     negative_concat = []
#     for i in range(len(data)):
#         if label[i] == 0:
#             positive.append(data[i])
#             positive_concat.append("".join([str(x) for x in data[i]]))
#         else:
#             negative.append(data[i])
#             negative_concat.append("".join([str(x) for x in data[i]]))