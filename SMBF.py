# 这个版本考虑使用多个哈希函数,且每个键的哈希函数族不一致
import mmh3

class MultiKeyBloomFilterWithSameHash:
    # dataset为多键的list，即一个二维数组
    def __init__(self, dataset, filtersize, k):
        self.dataset = dataset
        self.filtersize = filtersize
        self.fhi = [0 for _ in range(self.filtersize)]
        self.k = k
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash64(str(data[i]), seed=seed_id)[0] % self.filtersize
                self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash64(str(data[i]), seed=seed_id)[0] % self.filtersize
                if self.fhi[loc] == 0:
                    return False
        return True


# IMLBF-----backup
class IntervalMultiKeyBloomFilterWithDifferentHash:
    # dataset为多键的list，即一个二维数组
    def __init__(self, dataset, D, filtersize, k):
        #print(D)
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        self.k_list=[int(max(k+dff,2)) for dff in D]
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k_list[i]):
                loc = mmh3.hash64(str(data[i]), seed=seed_id + i * self.k_list[i])[0] % self.filtersize
                self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k_list[i]):
                loc = mmh3.hash64(str(data[i]), seed=seed_id + i * self.k_list[i])[0] % self.filtersize
                if self.fhi[loc] == 0:
                    return False
        return True
