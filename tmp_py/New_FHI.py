import pandas as pd
import mmh3
import hashlib


# 这个版本考虑对trigram哈希
class SimpleFHI:
    character = 26 + 26 + 10

    # dataset为string的list
    def __init__(self, dataset, n_gram=4, filtersize=88000):
        self.n_gram = n_gram
        self.dataset = dataset
        self.fhi = [0 for _ in range(filtersize)]
        self.filtersize = filtersize
        for x in self.dataset:
            self.insert(x)

    # 将str变为ngram形式，例如“abcd”->['a','b','c'] and ['b','c','d']
    def string_to_ngram(self, str):
        res = []
        if len(str) < self.n_gram:
            res.append(list(str))
            return res
        for i in range(0, len(str) - self.n_gram, 5):
            res.append(list(str[i:i + self.n_gram]))
        return res

    # 插入元素
    def insert(self, data):
        char_list = list(filter(str.isalnum, data))
        for gram in self.string_to_ngram(char_list):
            loc = mmh3.hash("".join(gram)) % self.filtersize
            self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        char_list = list(filter(str.isalnum, data))
        for gram in self.string_to_ngram(char_list):
            loc = mmh3.hash("".join(gram)) % self.filtersize
            if self.fhi[loc] == 0:
                return False
        return True


if __name__ == '__main__':
    ans = 1
    tau = 0.6
    interval = 4
    filter_size = 250000
    k = 7
    sample_num = 1000
    df = pd.read_csv("../file/file_score.csv")
    label = df["label"]
    score = df["score"]
    m = hashlib.md5()
    # m.update("dafstr".encode("utf8"))
    # print(m.hexdigest()[:5])
    for r1 in range(1, 10):
        for r2 in range(1, 10):
            md5_size = r1
            n_gram = r2
            positive_concat = []
            negative_concat = []
            for i in range(df.shape[0]):
                if label[i] == 0:
                    temp = []
                    for x in df.iloc[i, :-2]:
                        m.update(str(x).encode("utf8"))
                        temp.append(m.hexdigest()[:md5_size])
                    positive_concat.append("".join(temp))
                else:
                    temp = []
                    for x in df.iloc[i, :-2]:
                        m.update(str(x).encode("utf8"))
                        temp.append(m.hexdigest()[:md5_size])
                    negative_concat.append("".join(temp))
            df = pd.read_csv("../file/file_score.csv")
            # print(list(df.iloc[1, :]))
            fhi = SimpleFHI(positive_concat, filtersize=filter_size, n_gram=n_gram)
            count = 0
            for i in range(len(negative_concat)):
                if fhi.query(negative_concat[i]):
                    count += 1
            print("FPR", count / (len(negative_concat)), n_gram, md5_size)
