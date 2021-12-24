import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import random
from data.dataset_others import CriteoDataset


Model_Path_imdb = '../models_saved/dnn.pt'
FILE_PATH = '../lists_dicts/'
FILES = [i for i in os.listdir(FILE_PATH) if re.match(r'.*txt', i)]
FILES.remove('feature_sizes.txt')
FILES.remove('label.txt')
dimension_dic = {'date_year.txt': 1,'actor.txt': 2,'date_month.txt': 1, 'date_day.txt': 1, 'genre.txt': 2,
                 'duration.txt': 1, 'country.txt': 2, 'director.txt': 2, 'writer.txt': 2, 'company.txt': 2,
                'vote.txt': 1,'review.txt': 1, 'critic.txt': 1, 'label.txt': 1}
list_dic = {'date_year.txt': [],'actor.txt': [],'date_month.txt': [], 'date_day.txt': [], 'genre.txt': [],
            'duration.txt': [],'country.txt': [], 'director.txt': [], 'writer.txt': [], 'company.txt': [],
            'vote.txt': [],'review.txt': [], 'critic.txt': []}
number_files = ['date_year.txt', 'date_month.txt', 'date_day.txt', 'duration.txt', 'vote.txt',
                'review.txt', 'critic.txt']
categorial_files = ['actor.txt', 'genre.txt', 'country.txt', 'director.txt', 'writer.txt', 'company.txt']

def str_to_list(str_need, str_dimension):
    str_list = []
    if str_dimension == 1:
        try:
            str_list.append(int(str_need))
        except:
            str_list.append(float(str_need))
    elif str_dimension == 2:
        str_splited = str_need.split(',')
        for value in str_splited:
            str_list.append(int(value))
    elif str_dimension == 3:
        str_splited = str_need.strip('[').strip(']').split('],[')
        for vector_value in str_splited:
            word_vector = []
            vector_value_splited = vector_value.split(',')
            for value in vector_value_splited:
                word_vector.append(float(value))
            str_list.append(word_vector)

    return str_list

def read_txt(file_name, PATH = '../lists_dicts/'):
    data_list = []
    dimension_file = dimension_dic[file_name]
    with open(PATH+file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line_list = str_to_list(line, dimension_file)
            data_list.append(line_list)
    return data_list

def quchong(list_repeat):
    list_no_repeat = []
    for i in list_repeat:
        if i not in list_no_repeat:
            list_no_repeat.append(i)
    for i in range(len(list_no_repeat)):
        list_no_repeat[i] = list_no_repeat[i]+1
    return list_no_repeat

def find_max(list_max):
    length_max = 0
    for i in list_max:
        if len(i) > length_max:
            length_max = len(i)
    return length_max

def preprocess_imdb(percentage = 0.8):
        max_length_s = {'actor.txt': 0, 'genre.txt': 0, 'country.txt': 0,
                        'director.txt': 0, 'writer.txt': 0, 'company.txt': 0}

        for file in FILES:
            data_list = read_txt(file)
            if file in categorial_files:
                data_list_norepeat = []
                for value_list in data_list:
                    data_list_norepeat.append(quchong(value_list))
                max_length = find_max(data_list_norepeat)
                max_length_s[file] = max_length
                data_list = data_list_norepeat
            list_dic[file] = data_list
        labels = torch.from_numpy(np.loadtxt('../lists_dicts/label.txt'))
        print(labels)
        print(max_length_s)
        for file in categorial_files:
            list_dic[file] = pad_sequence([torch.from_numpy(np.array(x)) for x in list_dic[file]]
                                          , batch_first=True)
        for file in number_files:
            list_dic[file] = torch.from_numpy(np.array(list_dic[file]).reshape(-1, 1))

        xi_duration = list_dic['duration.txt']
        xi_vote = list_dic['vote.txt']
        xi_review = list_dic['review.txt']
        xi_critic = list_dic['critic.txt']
        xi_year = list_dic['date_year.txt']
        xi_month = list_dic['date_month.txt']
        xi_day = list_dic['date_day.txt']
        xi_genre = list_dic['genre.txt']
        xi_country = list_dic['country.txt']
        xi_company = list_dic['company.txt']
        xi_director = list_dic['director.txt']
        xi_writer = list_dic['writer.txt']
        xi_actor = list_dic['actor.txt']
        Xi = torch.cat([xi_duration, xi_vote, xi_review, xi_critic, xi_year, xi_month,
                        xi_day, xi_genre, xi_country, xi_company, xi_director, xi_writer,
                        xi_actor], dim=1)
        print(Xi[0])

        print(Xi.size(0), Xi.size(1), labels.size(0))
        length_data = labels.size(0)
        train_end = int(percentage * length_data)
        Xi_train = Xi[:train_end, :]
        labels_train = labels[:train_end]
        Xi_test = Xi[train_end:, :]
        labels_test = labels[train_end:]

        train_end_new = int(train_end * 0.9)
        Xi_train_new = Xi_train[:train_end_new, :]
        labels_train_new = labels_train[:train_end_new]
        Xi_val = Xi_train[train_end_new:, :]
        labels_val = labels_train[train_end_new:]

        Xi_train = Xi_train_new
        labels_train = labels_train_new
        num_pos = 0
        num_neg = 0
        for i in range(labels_train.size(0)):
            if labels_train[i] == 1:
                num_pos = num_pos + 1
            else:
                num_neg = num_neg + 1
        print(num_neg, num_pos)
        return Xi_train, labels_train, Xi_val, labels_val, Xi_test, labels_test, Xi, labels

class DNN(nn.Module):
    def __init__(self,  feature_sizes, feature_sections_dict = None, feature_sizes_dict = None,
                 dim_embedding = 256, dim_in = 45, dim_out = 1):
        super(DNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_embedding = dim_embedding
        self.feature_sizes = feature_sizes
        self.feature_sections_dict = feature_sections_dict
        self.feature_sizes_dict = feature_sizes_dict
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.dim_embedding) for feature_size in self.feature_sizes])
        self.linear_1 = nn.Linear(dim_embedding, 32)
        self.linear_2 = nn.Linear(32, 1)


    def forward(self, src):
        querys_tensor = []
        if self.feature_sections_dict != None:
            for k in range(src.size(0)):
                query_tensor = []
                for i, emb in enumerate(self.embeddings):
                    title = self.feature_sizes_dict[i]
                    feature_section = self.feature_sections_dict[title]
                    for j in range(feature_section[0], feature_section[1]):
                        a = src[k, j]
                        query_tensor.append(emb(a))
                b = torch.cat(query_tensor, dim=0)
                querys_tensor.append(b)
        else:
            for i in range(src.size(0)):
                query_tensor = []
                for j, emb in enumerate(self.embeddings):
                    a = src[i, j]
                    query_tensor.append(emb(a))
                b = torch.cat(query_tensor, dim=0)
                querys_tensor.append(b)
        src_embedded = torch.cat(querys_tensor).view(-1,self.dim_in,self.dim_embedding)
        #print(src_embedded.size())
        result = self.linear_1(src_embedded)
        result = self.linear_2(result)
        result = torch.sum(result,dim=1)
        return result


def check_accuracy(loader, model, if_train=True):
    print('>>>> checking...')
    if if_train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        result = []
        for x, y in loader:
            x = x.cuda()  # move to device, e.g. GPU
            y = y.float().cuda()
            total = model(x)
            preds = (F.sigmoid(total) > 0.5).squeeze(1)
            result.extend(list(np.array(F.sigmoid(total).cpu())))
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return result

def train(model, model_path, loader_train, loader_val, iter_time=100, use_gpu=True):
    print('>>>> training...')
    loss_function = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if (use_gpu):
        model = model.cuda()

    for _ in range(iter_time):
        model.train()
        for t, (x, y) in enumerate(loader_train):
            if (use_gpu):
                x = x.cuda().detach()
                y = y.float().unsqueeze(1).cuda().detach()
            y_pred = model(x)
            #print(y_pred)
            loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 500 == 0:
                acc = check_accuracy(loader_val, model)
    torch.save(model.state_dict(), model_path)
    return True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    batch_size = 100

    # 900000 items for training, 10000 items for valid, of all 1000000 items
    Num_train = 50000
    Num_test = 5000
    # load data
    train_data = CriteoDataset('../data', train=True)
    loader_train = DataLoader(train_data, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(Num_train)))
    val_data = CriteoDataset('../data', train=True)
    loader_val = DataLoader(val_data, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(Num_train, 55000)))

    test_data = CriteoDataset('../data', train=True)
    loader_test = DataLoader(val_data, batch_size=batch_size,
                             sampler=sampler.SubsetRandomSampler(range(55000, 60000)))

    feature_sizes = np.loadtxt('../data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print(feature_sizes)

    #dnn_model = DNN(feature_sizes,dim_in = 39)
    #train(dnn_model, Model_Path_imdb, loader_train, loader_val, iter_time = 100)
    test_model = DNN(feature_sizes,dim_in = 39).cuda()
    test_model.load_state_dict(torch.load(Model_Path_imdb))
    check_accuracy(loader_test, test_model, False)
    '''
    Xi_train, labels_train, Xi_val, labels_val, Xi_test, labels_test, Xi, labels \
        = preprocess_imdb()

    data_train = Data.TensorDataset(Xi_train, labels_train)
    data_val = Data.TensorDataset(Xi_val, labels_val)
    data_test = Data.TensorDataset(Xi_test, labels_test)
    data_all = Data.TensorDataset(Xi, labels)
    loader_train = Data.DataLoader(
        dataset=data_train,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    loader_val = Data.DataLoader(
        dataset=data_val,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    loader_test = Data.DataLoader(
        dataset=data_test,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    loader_all = Data.DataLoader(
        dataset=data_all,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    feature_sizes = np.loadtxt('../lists_dicts/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    feature_sections_dict = {'duration.txt': [0, 1], 'vote.txt': [1, 2], 'review.txt': [2, 3], 'critic.txt': [3, 4],
                             'date_year.txt': [4, 5], 'date_month.txt': [5, 6], 'date_day.txt': [6, 7],
                             'genre.txt': [7, 10], 'country.txt': [10, 24],
                             'company.txt': [24, 26], 'director.txt': [26, 28], 'writer.txt': [28, 30],
                             'actor.txt': [30, 45]}
    feature_sizes_dict = {0: 'duration.txt', 1: 'vote.txt', 2: 'review.txt', 3: 'critic.txt', 4: 'date_year.txt',
                          5: 'date_month.txt', 6: 'date_day.txt', 7: 'genre.txt', 8: 'country.txt', 9: 'director.txt',
                          10: 'writer.txt',
                          11: 'company.txt', 12: 'actor.txt'}
    dnn_model = DNN(feature_sizes, feature_sections_dict, feature_sizes_dict, dim_embedding = 256, dim_in = 45, dim_out = 1)
    train(dnn_model, Model_Path_imdb, loader_train, loader_val, iter_time = 100)
    check_accuracy(loader_test, dnn_model, False)
    result = check_accuracy(loader_all, dnn_model, False)
    df_result = pd.DataFrame(np.array(result).reshape(-1, 1))
    df_result.to_csv('dnn_imdb.csv')
    '''

