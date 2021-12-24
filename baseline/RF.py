import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle

Model_Path_imdb = '../models_saved/rf.pt'
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
    return np.array(Xi_train), np.array(labels_train), np.array(Xi_val), np.array(labels_val), \
           np.array(Xi_test), np.array(labels_test), np.array(Xi), np.array(labels)

def preprocess_kaggls(percentage = 0.8):
    df_data = pd.read_csv(os.path.join('../data', 'train.txt'))
    data = df_data.iloc[:, :-1].values
    labels = df_data.iloc[:, -1].values
    return data, labels

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
        for x, y in loader:
            x = x.float().cuda()  # move to device, e.g. GPU
            y = y.float().cuda()
            total = model(x)
            preds = (F.sigmoid(total) > 0.5).squeeze(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return acc

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
                x = x.float().cuda().detach()
                y = y.float().unsqueeze(1).cuda().detach()
            y_pred = model(x)
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

    Xi_train, labels_train, Xi_val, labels_val, Xi_test, labels_test, Xi, labels \
        = preprocess_imdb()

    #Xi, labels=preprocess_kaggls()
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    Xi = scaler_for_x.fit_transform(Xi)
    X_train, X_test, Y_train, Y_test = train_test_split(Xi, labels, test_size=0.7, random_state=7)

    clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=8)
    clf.fit(X_train, Y_train)
    with open('../models_saved/RF.pickle', 'wb') as f:
        pickle.dump(clf, f)
    score = []
    with open('../models_saved/RF.pickle', 'rb') as f:
        clf_test = pickle.load(f)
        predict_result = clf_test.predict(X_test)
        predict_pb = clf_test.predict_proba(X_test)
        if len(predict_result) != len(Y_test):
            print("error!")
        num = 0
        for i in range(len(predict_result)):
            if predict_result[i] == Y_test[i]:
                num = num + 1
        ac = num / len(predict_result)
        print(predict_result)
        print(predict_pb)
        for i in range(len(predict_result)):
            score_value = predict_pb[i][1]
            score.append(score_value)
        for i in range(len(predict_result)):
            if Y_test[i] == 0:
                Y_test[i] = -1

        print(score)
        print(ac)