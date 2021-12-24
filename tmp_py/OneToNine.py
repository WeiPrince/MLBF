import random

import pandas as pd


def toNum(x):
    a = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99]
    for i in range(8):
        if x <= a[i]:
            return i
    return 8


def getindex(x, list):
    l = 0
    r = len(list) - 1
    while (l <= r):
        mid = (l + r) // 2
        if x > list[mid]:
            l = mid + 1
        elif x < list[mid]:
            r = mid - 1
        else:
            return mid
    print("未找到")
    return -1


# 取第3，9，10，11列进行划分
col = [2, 8, 9, 10]
df = pd.read_csv("../file/OneToNine.csv")
positive = df[df["label"] == 1]
# print(df.shape[0])
# for i in range(df.shape[0]):
#     if df.iloc[i, -1] == 1:
#         continue
#     for k in col:
#         positive_size = positive.shape[0]
#         random_idx = random.randint(0, positive_size - 1)
#         df.iloc[i, k] = positive.iloc[random_idx, k]
# print("负例生成完成")
# df.to_csv("OneToNine.csv", index=False)
print("开始映射到0~8")
for i in col:
    print("正在处理第", i, "行")
    templist = sorted(df.iloc[:, i])
    len1 = len(templist)
    for j in range(len1):
        loc = getindex(df.iloc[j, i], templist)
        if loc == -1:
            print("错误")
        df.iloc[j, i] = toNum(loc / len1)
df.to_csv("OneToNine_output.csv", index=False)
