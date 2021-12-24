import pandas as pd
import mmh3
import numpy as np
import matplotlib.pyplot as plt


# 这个版本考虑使用多个哈希函数
class SimpleFHI:
    # dataset为string的list
    def __init__(self, dataset, filtersize, k):
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        self.k = k
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string类型
    def insert(self, data):
        for seed_id in range(self.k):
            loc = mmh3.hash(data, seed=seed_id) % self.filtersize
            self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for seed_id in range(self.k):
            loc = mmh3.hash(data, seed=seed_id) % self.filtersize
            if self.fhi[loc] == 0:
                return False
        return True


# 这个版本考虑使用多个哈希函数,且每个键的哈希函数族不一致
class MultiKeyBloomFilterWithDifferentHash:
    # dataset为多键的list，即一个二维数组
    def __init__(self, dataset, filtersize=50000, k=5):
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        self.k = k
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id + i * self.k) % self.filtersize
                self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id + i * self.k) % self.filtersize
                if self.fhi[loc] == 0:
                    return False
        return True


# 这个版本考虑使用多个哈希函数,且每个键的哈希函数族不一致
class MultiKeyBloomFilterWithDifferentHash_new:
    # dataset为多键的list，即一个二维数组
    def __init__(self, dataset, std, filtersize=50000, k=5):
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        self.k = k
        self.std = std
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            if self.std[i] <= 10:
                self.k = 3
            elif 10 < self.std[i] <= 100:
                self.k = 4
            else:
                self.k = 5
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id + i * self.k) % self.filtersize
                self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            if self.std[i] <= 10:
                self.k = 3
            elif 10 < self.std[i] <= 100:
                self.k = 4
            else:
                self.k = 5
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id + i * self.k) % self.filtersize
                if self.fhi[loc] == 0:
                    return False
        return True


# 这个版本考虑使用多个哈希函数，且每个哈希函数族一致
class MultiKeyBloomFilterWithSameHash:
    # dataset为多键的list，即一个二维数组
    def __init__(self, dataset, filtersize=50000, k=5):
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        self.k = k
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id) % self.filtersize
                self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash(str(data[i]), seed=seed_id) % self.filtersize
                if self.fhi[loc] == 0:
                    return False
        return True


if __name__ == '__main__':
    filter_size = 130003
    insert_size = 18000
    query_size = 2000
    df = pd.read_csv("file/file.csv")
    # print(list(df.iloc[1, :]))
    print("将多个键值拼接")
    dataset = ["".join([str(e) for e in list(df.iloc[i, :])]) for i in range(20000)]
    fhi = SimpleFHI(dataset[0:insert_size])
    print(sum(fhi.fhi))
    count = 0
    fnr = 0
    for i in range(insert_size):
        if fhi.query(dataset[i]):
            fnr += 1
    print(fnr)
    for i in range(query_size):
        if fhi.query(dataset[insert_size + i]):
            count += 1
    print("FPR", count / query_size)
    print("method1:不拼接多个键值，而是采用多个BLoom Filter")
    filters = []
    for i in range(df.shape[1]):
        d = [str(e) for e in df.iloc[:insert_size, i]]
        f = SimpleFHI(d, filtersize=filter_size // df.shape[1], k=5)
        filters.append(f)
    query_count = 0
    for i in range(query_size):
        tag = True
        for j in range(df.shape[1]):
            query_d = str(df.iloc[insert_size + i, j])
            if not filters[j].query(query_d):
                tag = False
                break
        if tag:
            query_count += 1
    print("FPR:", query_count / query_size)

    print("********************")
    print("method2:使用一个bloom filter，且每个键值采用不同的哈希函数")
    df = np.array(pd.read_csv("file/file.csv")).tolist()
    MKBF = MultiKeyBloomFilterWithDifferentHash(df[0:insert_size], filtersize=filter_size, k=4)
    c = 0
    for i in range(query_size):
        if MKBF.query(df[insert_size + i]):
            c += 1;
    print("FPR:", c / query_size)

    print("********************")
    print("method3:使用一个bloom filter，且每个键值采用相同的哈希函数")
    df = np.array(pd.read_csv("file/file.csv")).tolist()
    MKBF = MultiKeyBloomFilterWithSameHash(df[0:insert_size], filtersize=filter_size)
    c = 0
    for i in range(query_size):
        if MKBF.query(df[insert_size + i]):
            c += 1
    print("FPR:", c / query_size)
    std = []
    # 统计哈希结果
    for i in range(len(list(pd.read_csv("file/file.csv")))):
        d = [str(e) for e in df[:insert_size][i]]
        hash_res = []
        for j in range(insert_size):
            hash_res.append(mmh3.hash(str(df[j][i])) % filter_size)

        plt_num = [0 for j in range(filter_size // 50 + 1)]
        for r in hash_res:
            plt_num[r // 50] += 1
        # plt_num = sorted(plt_num)
        std.append(np.std(np.array(plt_num)))
        print("第" + str(i + 1) + "个数据的方差为：" + str(np.std(np.array(plt_num))))
        plt.bar(range(len(plt_num)), plt_num)
        plt.xlabel(list(pd.read_csv("file/file.csv"))[i])
        plt.show()
    print("********************")
    print("method4:使用一个bloom filter，且每个键值采用不同的哈希函数,改进哈希函数的个数，即不为恒定的k=5")
    df = np.array(pd.read_csv("file/file.csv")).tolist()
    MKBF = MultiKeyBloomFilterWithDifferentHash_new(df[0:insert_size], std, filtersize=filter_size)
    c = 0
    for i in range(query_size):
        if MKBF.query(df[insert_size + i]):
            c += 1
    print("FPR:", c / query_size)
