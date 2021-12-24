import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import os
import re
import math

torch.manual_seed(1)

BATCH_SIZE = 100

POSITIVE_PATH = 'lists_dicts/'
FILES_POSITIVE = [i for i in os.listdir(POSITIVE_PATH) if re.match(r'.*txt', i)]
FILES_POSITIVE.remove('feature_sizes.txt')
FILES_POSITIVE.remove('label.txt')

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

def read_txt(file_name, PATH = 'lists_dicts/'):
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

def preprocess(percentage = 0.3):
    max_length_s = {'actor.txt': 0, 'genre.txt': 0, 'country.txt': 0,
                    'director.txt': 0, 'writer.txt': 0, 'company.txt': 0}

    for file in FILES_POSITIVE:
        data_list = read_txt(file)
        if file in categorial_files:
            data_list_norepeat = []
            for value_list in data_list:
                data_list_norepeat.append(quchong(value_list))
            max_length = find_max(data_list_norepeat)
            max_length_s[file] = max_length
            data_list = data_list_norepeat
        list_dic[file] = data_list

    labels = torch.from_numpy(np.loadtxt('lists_dicts/label.txt'))
    print(labels)
    print(max_length_s)

    for file in categorial_files:
        list_dic[file] = pad_sequence([torch.from_numpy(np.array(x)) for x in list_dic[file]]
                                      , batch_first=True)
    for file in number_files:
        list_dic[file] = torch.from_numpy(np.array(list_dic[file]).reshape(-1,1))

    length = list_dic['date_year.txt'].size(0)
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
    Xi = torch.cat([xi_duration, xi_vote, xi_review, xi_critic,xi_year,xi_month,
                    xi_day,xi_genre,xi_country,xi_company,xi_director,xi_writer,
                    xi_actor], dim=1)
    print(Xi[0])
    xv_duration = torch.ones(length, 1)
    xv_vote = torch.ones(length, 1)
    xv_review = torch.ones(length, 1)
    xv_critic = torch.ones(length, 1)
    xv_year = torch.ones(length, 1)
    xv_month = torch.ones(length, 1)
    xv_day = torch.ones(length, 1)
    a = np.zeros((list_dic['genre.txt'].size(0),list_dic['genre.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['genre.txt']):
            if list_dic['genre.txt'][i][j] != 0:
                a[i][j] = 1
    xv_genre = torch.from_numpy(a)
    a = np.zeros((list_dic['country.txt'].size(0), list_dic['country.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['country.txt']):
            if list_dic['country.txt'][i][j] != 0:
                a[i][j] = 1
    xv_country = torch.from_numpy(a)
    a = np.zeros((list_dic['company.txt'].size(0), list_dic['company.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['company.txt']):
            if list_dic['company.txt'][i][j] != 0:
                a[i][j] = 1
    xv_company = torch.from_numpy(a)
    a = np.zeros((list_dic['director.txt'].size(0), list_dic['director.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['director.txt']):
            if list_dic['director.txt'][i][j] != 0:
                a[i][j] = 1
    xv_director = torch.from_numpy(a)
    a = np.zeros((list_dic['writer.txt'].size(0), list_dic['writer.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['writer.txt']):
            if list_dic['writer.txt'][i][j] != 0:
                a[i][j] = 1
    xv_writer = torch.from_numpy(a)
    a = np.zeros((list_dic['actor.txt'].size(0), list_dic['actor.txt'].size(1)))
    for i in range(length):
        for j in range(max_length_s['actor.txt']):
            if list_dic['actor.txt'][i][j] != 0:
                a[i][j] = 1
    xv_actor = torch.from_numpy(a)
    Xv = torch.cat([xv_duration, xv_vote, xv_review, xv_critic, xv_year, xv_month,
                    xv_day, xv_genre, xv_country, xv_company, xv_director, xv_writer,
                    xv_actor], dim=1)


    print(Xi.size(0),Xi.size(1),Xv.size(0),Xv.size(1),labels.size(0))
    length_data = labels.size(0)
    train_end = int(percentage * length_data)
    Xi_train = Xi[:train_end, :]
    Xv_train = Xv[:train_end, :]
    labels_train = labels[:train_end]
    Xi_test = Xi[train_end:, :]
    Xv_test = Xv[train_end:, :]
    labels_test = labels[train_end:]

    train_end_new = int(train_end*0.9)
    Xi_train_new = Xi_train[:train_end_new, :]
    Xv_train_new = Xv_train[:train_end_new, :]
    labels_train_new = labels_train[:train_end_new]
    Xi_val = Xi_train[train_end_new:, :]
    Xv_val = Xv_train[train_end_new:, :]
    labels_val = labels_train[train_end_new:]

    Xi_train = Xi_train_new
    Xv_train = Xv_train_new
    labels_train = labels_train_new
    num_pos = 0
    num_neg = 0
    for i in range(labels_train.size(0)):
        if labels_train[i] == 1:
            num_pos = num_pos + 1
        else:
            num_neg = num_neg + 1
    print(num_neg,num_pos)
    return Xi_train,Xv_train,labels_train,Xi_val,Xv_val,labels_val,Xi_test,Xv_test,labels_test,Xi,Xv,labels


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=256,
                 hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.randn(1))
        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        """
            init deep part
        """
        all_dims = [45 * self.embedding_size] + \
                   self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))
        self.sigmod = nn.Sigmoid()
    def forward(self, Xi, Xv):
        """
        Forward process of network.

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, s
        hape of (N, field_size, 1)
        """
        """
            fm part
        """
        feature_sections_dict = {'duration.txt':[0,1],'vote.txt':[1,2],
        'review.txt':[2,3],'critic.txt':[3,4],'date_year.txt':[4,5],'date_month.txt':[5,6],
        'date_day.txt':[6,7],'genre.txt':[7,10],'country.txt':[10,24],'company.txt':[24,26],
        'director.txt':[26,28],'writer.txt':[28,30],'actor.txt':[30,45]}

        feature_sizes_dict = {0:'duration.txt',1:'vote.txt',2:'review.txt',
        3:'critic.txt',4:'date_year.txt',5:'date_month.txt',6:'date_day.txt',7:'genre.txt',
        8:'country.txt',9:'director.txt',10:'writer.txt',11:'company.txt',12:'actor.txt'}

        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_embeddings):
            title = feature_sizes_dict[i]
            #print(title)
            feature_section = feature_sections_dict[title]
            for j in range(feature_section[0],feature_section[1]):
                a = torch.unsqueeze(Xi[:,j], 1)
                #print(a.max()),print(j)
                b = (torch.sum(emb(a),1).t() * Xv[:,j]).t()
                fm_first_order_emb_arr.append(b)
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_embeddings):
            title = feature_sizes_dict[i]
            feature_section = feature_sections_dict[title]
            for j in range(feature_section[0],feature_section[1]):
                a = torch.unsqueeze(Xi[:,j], 1)
                b = (torch.sum(emb(a),1).t() * Xv[:,j]).t()
                fm_second_order_emb_arr.append(b)
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        #print(fm_sum_second_order_emb)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
            fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item*item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        #print(total_sum)
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations.
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits
        #criterion = nn.BCELoss()

        for _ in range(epochs):
            for t, (x, y) in enumerate(loader_train):
                xi = x[:,:45]
                xv = x[:,45:]
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
                total = model(xi, xv)
                total_sigmod = F.sigmoid(total)

                weight = torch.ones(y.size(0))
                for i in range(y.size(0)):
                    if y[i] == 0:
                        weight[i] = math.exp( 2 * total_sigmod[i] )

                loss = criterion(total, y, weight = weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self.check_accuracy(loader_val, model, True)

    def check_accuracy(self, loader, model, if_train=True):
        if if_train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                xi = x[:, :45]
                xv = x[:, 45:]
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.bool)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
        return F.sigmoid(total)

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    Xi_train,Xv_train,labels_train,Xi_val,Xv_val,labels_val,Xi_test,Xv_test,labels_test,Xi,Xv,labels = preprocess()

    X_train = torch.cat([Xi_train,Xv_train],1)
    X_val = torch.cat([Xi_val,Xv_val],1)
    data_train = Data.TensorDataset(X_train, labels_train)
    data_val = Data.TensorDataset(X_val, labels_val)
    loader_train = Data.DataLoader(
        dataset=data_train,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=BATCH_SIZE,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    loader_val = Data.DataLoader(
        dataset=data_val,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=BATCH_SIZE,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    feature_sizes = np.loadtxt('lists_dicts/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print(feature_sizes)
    model = DeepFM(feature_sizes, use_cuda=True)
    optimizer = optim.Adam(model.parameters(), lr=4e-5, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=50, verbose=True)
    torch.save(model.state_dict(), 'models_saved/DeepFM.pt')
    X_test = torch.cat([Xi_test, Xv_test], 1)
    data_test = Data.TensorDataset(X_test, labels_test)
    loader_test = Data.DataLoader(
        dataset=data_test,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=BATCH_SIZE,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    model.check_accuracy(loader_test, model, False)
    X_all = torch.cat([Xi, Xv], 1)
    data_all = Data.TensorDataset(X_all, labels)
    '''
    batch_all = labels.size(0)
    loader_all = Data.DataLoader(
        dataset=data_all,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_all,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    result = model.check_accuracy(loader_all, model, False)
    df_result = pd.DataFrame(np.array(result).reshape(-1,1))
    df_result.to_csv('possibility_imdb.csv')
    '''





