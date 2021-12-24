import f as f
import numpy as np
import pandas as pd
import SMBF
import random

f = open("train.txt", "r")  # 设置文件对象
FP_size = 1200
negative_size = 100000
positive_size = 100000
positive_data = []
negative_data = []
c = 0
while True:
    line = f.readline().strip("\n").split("\t")
    l = int(line[0])
    if l == 1:
        positive_size -= 1
        positive_data.append(line)
    if positive_size == 0:
        break
positive_data = np.array(positive_data)
df = pd.DataFrame(positive_data)
df.to_csv("CTR_positive.csv", index=False)
print("正例生成完成")
while True:
    line = f.readline().strip("\n").split("\t")
    l = int(line[0])
    c += 1
    if c % 1000 == 0:
        print("正在读取第", c, "条, 生成条数: ", len(negative_data))
    if l == 0:
        neg = line
        unexist = False
        count = 0
        for i in range(1, len(neg)):
            if str(neg[i]) not in [str(xx) for xx in positive_data[:, i]]:
                # print(i, len(neg), neg[i], positive_data[:, i])
                unexist = True
                break
        if unexist:
            if len(negative_data) % 1000 == 0:
                print("生成条数: ", len(negative_data))
            negative_data.append(line)
            negative_size -= 1
        elif FP_size > 0:
            if len(negative_data) % 1000 == 0:
                print("生成条数: ", len(negative_data))
            negative_data.append(line)
            FP_size -= 1
    if negative_size == 0:
        print("负例数据已经生成完成, 其中FP_size:", FP_size)
        break
datafram = pd.DataFrame(negative_data)
datafram.to_csv("CTR_negative.csv", index=False)
print("负例生成完成")

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
