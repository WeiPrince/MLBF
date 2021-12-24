import pandas as pd


class SimpleFHI:
    character = 26 + 26 + 10

    # dataset为string的list
    def __init__(self, dataset, n_gram=3):
        self.n_gram = n_gram
        self.dataset = dataset
        self.fhi = [0 for _ in range(self.character ** self.n_gram + 10)]
        for x in self.dataset:
            self.insert(x)

    # 将str变为ngram形式，例如“abcd”->['a','b','c'] and ['b','c','d']
    def string_to_ngram(self, str):
        res = []
        if len(str) < self.n_gram:
            res.append(list(str))
            return res
        for i in range(len(str) - self.n_gram):
            res.append(list(str[i:i + self.n_gram]))
        return res

    # 返回字符对应的数字
    def get_char_num(self):
        char_num = dict()
        for i in range(48, 58):  # 0-9
            char_num[chr(i)] = i - 48
        for i in range(97, 123):  # a-z
            char_num[chr(i)] = i - 97 + 10
        for i in range(65, 91):  # A-Z
            char_num[chr(i)] = i - 65 + 26 + 10
        return char_num

    # 插入元素
    def insert(self, data):
        char_list = list(filter(str.isalnum, data))
        for gram in self.string_to_ngram(char_list):
            loc = 0
            for c in gram:
                if ord(c) not in range(48, 58) and ord(c) not in range(97, 123) and ord(c) not in range(65, 91):
                    loc = loc * self.character
                    # loc = loc + self.get_char_num()[c]
                    continue
                loc = loc * self.character
                loc = loc + self.get_char_num()[c]
            # print(loc)
            self.fhi[loc] = 1

    # 如果data存在则返回True
    def query(self, data):
        char_list = list(filter(str.isalnum, data))
        for gram in self.string_to_ngram(char_list):
            loc = 0
            for c in gram:
                if ord(c) not in range(48, 58) and ord(c) not in range(97, 123) and ord(c) not in range(65, 91):
                    loc = loc * self.character
                    # loc = loc + self.get_char_num()[c]
                    continue
                loc = loc * self.character
                loc = loc + self.get_char_num()[c]
            if self.fhi[loc] == 0:
                return False
        return True


if __name__ == '__main__':
    df = pd.read_csv("../file/file.csv")
    # print(list(df.iloc[1, :]))
    dataset = ["".join([str(e) for e in list(df.iloc[i, :])]) for i in range(7000)]
    fhi = SimpleFHI(dataset[0:5000])
    print(sum(fhi.fhi))
    count = 0
    for i in range(1000):
        if fhi.query(dataset[5000 + i]):
            count += 1
    print(count / 1000)
