import mmh3


class MMBF:
    # dataset为string的list
    def __init__(self, dataset, filtersize, k):
        self.filtersize = filtersize // len(dataset[0])
        self.dataset = dataset
        self.fhi = [[0 for _ in range(self.filtersize)] for _ in range(len(dataset[0]))]
        self.k = k
        for x in self.dataset:
            self.insert(x)

    # 插入元素,data为string的list类型
    def insert(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash64(data[i], seed=seed_id)[0] % self.filtersize
                self.fhi[i][loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        for i in range(len(data)):
            for seed_id in range(self.k):
                loc = mmh3.hash64(data[i], seed=seed_id)[0] % self.filtersize
                if self.fhi[i][loc] == 0:
                    return False
        return True
