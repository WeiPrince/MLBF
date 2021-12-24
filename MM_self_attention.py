import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import os
import re
import math

Model_Path = 'models_saved/transformer_fake.pt'
POSITIVE_PATH = 'lists_dicts/'
FILES_POSITIVE = [i for i in os.listdir(POSITIVE_PATH) if re.match(r'.*txt', i)]
FILES_POSITIVE.remove('feature_sizes.txt')

dimension_dic = {'date_year.txt': 1,'actor.txt': 2,'date_month.txt': 1, 'date_day.txt': 1, 'genre.txt': 2,
                 'duration.txt': 1, 'country.txt': 2, 'director.txt': 2, 'writer.txt': 2, 'company.txt': 2,
                 'vote.txt': 1,'review.txt': 1, 'critic.txt': 1}
list_dic = {'date_year.txt': [],'actor.txt': [],'date_month.txt': [], 'date_day.txt': [], 'genre.txt': [],
            'duration.txt': [],'country.txt': [], 'director.txt': [], 'writer.txt': [], 'company.txt': [],
            'vote.txt': [],'review.txt': [], 'critic.txt': []}
number_files = ['date_year.txt', 'date_month.txt', 'date_day.txt', 'duration.txt', 'vote.txt',
                'review.txt', 'critic.txt']
categorial_files = ['actor.txt', 'genre.txt', 'country.txt', 'director.txt', 'writer.txt', 'company.txt']
TRAIN_NUM = 0

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.generator = generator

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        encoder_result = torch.sum(self.encode(src, src_mask), dim=1)
        #print(self.encode(src, src_mask).size())
        #encoder_result = self.encode(src, src_mask).view(-1,45 * 256)
        encoder_result =self.generator(encoder_result)
        return encoder_result

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, dim_out):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, dim_out)
        #self.sigmoid = torch.sigmoid()

    def forward(self, x):
        #result = torch.sigmoid(self.proj(x))
        result = self.proj(x)
        return result

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layers, layer_size):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm(layer_size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        #print(x.size())

        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #self.sublayer_1 = SublayerConnection(size, dropout)
        #self.sublayer_2 = SublayerConnection(size, dropout)
        self.norm_1 = LayerNorm(size)
        self.norm_2 = LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
       # self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = x + self.dropout_1(self.self_attn(self.norm_1(x), self.norm_1(x), self.norm_1(x), None))
        return x + self.dropout_2(self.feed_forward(self.norm_2(x)))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        #self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linear_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linear_o(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def make_model(d_model=256, d_out=1, d_ff=512, h=4, dropout=0.5):

    "Helper: Construct a model from hyperparameters."
    #c = copy.deepcopy
    attn_1 = MultiHeadedAttention(h, d_model)
    attn_2 = MultiHeadedAttention(h, d_model)

    ff_1 = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_2 = PositionwiseFeedForward(d_model, d_ff, dropout)

    Encoder_lists = [EncoderLayer(d_model, attn_1, ff_1, dropout),EncoderLayer(d_model, attn_2, ff_2, dropout)]
    Encoder_layers = nn.ModuleList(Encoder_lists)
    model = EncoderDecoder(
        Encoder(Encoder_layers,d_model),
        Generator(d_model, d_out))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def str_to_list(str_need,str_dimension):
    str_list = []
    if str_dimension == 1:
        try:
            str_list.append(int(str_need))
        except:
            str_list.append(float(str_need))
    elif str_dimension == 2:
        str_splited = str_need.split(',')
        for value in str_splited:
            value = value.strip()
            str_list.append(value)
    return str_list


def load_data(file_path = 'dataset_imdb/filtered_data_fake.csv',percentage = 0.3):
    dict_files = [i for i in os.listdir('dicts/') if re.match(r'.*csv', i)]
    dicts = {'actors':{},'genre':{},'country':{},'director':{},'writer':{},'production_company':{}}
    for f in dict_files:
        f_dict = open('dicts/'+f, encoding="utf-8")
        df_dict = pd.read_csv(f_dict)
        dict_keys = df_dict.columns.values
        dict = {k:df_dict.loc[0,k] for k in dict_keys}
        dicts[f[:-4]] = dict
    feature_sizes = np.loadtxt('lists_dicts/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    m_state_dict = torch.load('models_saved/DeepFM.pt')

    '''
        order: { xv_duration, xv_average, xv_vote, xv_review, xv_critic, xv_year, xv_month,
                    xv_day, xv_genre, xv_country, xv_company, xv_director, xv_writer,
                    xv_actor }
    '''
    pretrained_embedding_parameter= {int(k.strip('fm_second_order_embeddings.').strip('.weight')): v
                                 for k, v in m_state_dict.items() if 'second_order_embeddings' in k}
    pretrained_embedding_dict = nn.ModuleList(
        [nn.Embedding(feature_size, 256) for feature_size in feature_sizes])
    for i in range(len(pretrained_embedding_parameter)):
        pretrained_embedding_dict[i].load_state_dict({'weight':pretrained_embedding_parameter[i]})

    feature_order = {'duration':0,'votes':1,'reviews_from_users':2,'reviews_from_critics':3,'year':4,
                     'month':5,'day':6,'genre':7,'country':8,'director':9,'writer':10,'production_company':11,'actors':12}
    features_len = {'duration':1,'votes':1,'reviews_from_users':1,'reviews_from_critics':1,'year':1,
                    'month':1,'day':1,'actors':15,'genre':3,'country':14,'director':2,'writer':2,'production_company':2}
    f = open(file_path)
    df_query = pd.read_csv(f)
    #labels = torch.from_numpy(np.loadtxt('lists_dicts/label.txt').reshape(-1)).float()
    a = list(df_query.iloc[:, 11])
    for i in range(len(a)):
        a[i] = int(a[i])
    labels = torch.from_numpy(np.array(a)).float()
    df_query = df_query.drop(df_query.columns[[11]], axis=1)
    print(df_query)
    print(labels)
    columns = df_query.columns.values
    querys_tensor = []
    for i in range(len(df_query)):
        query_tensor = []
        for j in columns:
            if j =='date_published':
                if '/' in df_query.loc[i,j]:
                    date_list = df_query.loc[i,j].split('/')
                else:
                    date_list = df_query.loc[i, j].split('-')
                query_tensor.append(
                    pretrained_embedding_dict[feature_order['year']]
                    (torch.from_numpy(np.array([int(date_list[0])-1894]))))
                query_tensor.append(
                    pretrained_embedding_dict[feature_order['month']]
                    (torch.from_numpy(np.array([int(date_list[1])-1]))))
                query_tensor.append(
                    pretrained_embedding_dict[feature_order['day']]
                    (torch.from_numpy(np.array([int(date_list[2]) - 1]))))
            else:
                if j in ['duration','votes','reviews_from_users','reviews_from_critics']:
                    data_value = str_to_list(df_query.loc[i, j],1)
                    query_tensor.append(
                        pretrained_embedding_dict[feature_order[j]]
                        (torch.from_numpy(np.array(data_value))))
                else:
                    data_value = str_to_list(df_query.loc[i, j], 2)
                    data_list = data_value[:15]
                    for l in range(len(data_list)):
                        if data_list[l] in dicts[j].keys():
                            query_tensor.append(
                            pretrained_embedding_dict[feature_order[j]]
                            (torch.from_numpy(np.array([dicts[j][data_list[l]]]))))
                        else:
                            query_tensor.append(
                                pretrained_embedding_dict[feature_order[j]]
                                (torch.from_numpy(np.array([dicts[j]['<unk>']]))))
                    for _ in range(features_len[j]-len(data_list)):
                        query_tensor.append(
                            pretrained_embedding_dict[feature_order[j]]
                            (torch.from_numpy(np.array([0]))))
        tensor_of_query = torch.cat(query_tensor,dim = 0)
        querys_tensor.append(tensor_of_query)
    for i in range(len(querys_tensor)):
        if querys_tensor[i].size(0)!=45:
            print(i,querys_tensor[i].size())
            print(querys_tensor[i])
    '''
    querys_tensor_pos = []
    querys_tensor_neg = []
    for i in range(len(querys_tensor)):
        if labels[i] == 1:
            querys_tensor_pos.append(querys_tensor[i])
        else:
            querys_tensor_neg.append(querys_tensor[i])
    print(len(querys_tensor_pos),len(querys_tensor_neg))
    train_end = int(len(querys_tensor) * 0.5)
    train_end_pos = int(train_end*0.35)
    train_end_neg = int(train_end*0.65)
    data_train = querys_tensor_neg[:train_end_neg]+\
                 querys_tensor_pos[:train_end_pos]
    label_train_pos = np.ones(train_end_pos)
    label_train_neg = np.zeros(train_end_neg)
    label_train = torch.from_numpy(np.concatenate([label_train_neg,label_train_pos]))
    valid_end = int(len(querys_tensor) * 0.05)
    valid_end_pos = train_end_pos + int(valid_end * 0.5)
    valid_end_neg = train_end_neg + int(valid_end * 0.5)
    data_valid = querys_tensor_neg[train_end_neg:valid_end_neg]+\
               querys_tensor_pos[train_end_pos:valid_end_pos]
    label_valid_pos = np.ones(valid_end_pos - train_end_pos)
    label_valid_neg = np.zeros(valid_end_neg - train_end_neg)
    label_valid = torch.from_numpy(np.concatenate([label_valid_neg, label_valid_pos]))
    data_test = querys_tensor_neg[valid_end_neg:] + \
                 querys_tensor_pos[valid_end_pos:]
    label_test_pos = np.ones(len(querys_tensor_pos[valid_end_pos:]))
    label_test_neg = np.zeros(len(querys_tensor_neg[valid_end_neg:]))
    label_test = torch.from_numpy(np.concatenate([label_test_neg, label_test_pos]))

    size_0 = data_train[0].size(0)
    size_1 = data_train[0].size(1)
    length_train = len(data_train)
    length_valid = len(data_valid)
    length_test = len(data_test)
    print(length_train,length_valid,length_test)
    print(data_train[0].size(0),data_train[0].size(1))
    print(data_valid[0].size(0),data_valid[0].size(1))
    print(data_test[0].size(0),data_test[0].size(1))
    X_train = torch.cat(data_train).view(length_train,size_0,size_1)
    X_valid = torch.cat(data_valid).view(length_valid,size_0,size_1)
    X_test = torch.cat(data_test).view(length_test,size_0,size_1)
    X = torch.cat(querys_tensor).view(-1,size_0,size_1)
    num_pos = 0
    for i in range(len(label_train)):
        if label_train[i] ==1:
            num_pos = num_pos+1
    print(num_pos/len(label_train))
    '''
    length_data = len(querys_tensor)
    train_end = int(percentage * length_data)
    data_train = querys_tensor[:train_end]
    label_train = labels[:train_end]
    data_test = querys_tensor[train_end:]
    label_test = labels[train_end:]

    train_end_new = int(train_end * 0.9)
    data_train_new = data_train[:train_end_new]
    label_train_new = label_train[:train_end_new]
    data_valid = data_train[train_end_new:]
    label_valid = label_train[train_end_new:]
    data_train = data_train_new
    label_train = label_train_new

    size_0 = data_train[0].size(0)
    size_1 = data_train[0].size(1)
    length_train = len(data_train)
    length_valid = len(data_valid)
    length_test = len(data_test)
    print(length_train, length_valid, length_test)
    print(data_train[0].size(0), data_train[0].size(1))
    print(data_valid[0].size(0), data_valid[0].size(1))
    print(data_test[0].size(0), data_test[0].size(1))
    X_train = torch.cat(data_train).view(length_train, size_0, size_1)
    X_valid = torch.cat(data_valid).view(length_valid, size_0, size_1)
    X_test = torch.cat(data_test).view(length_test, size_0, size_1)
    X = torch.cat(querys_tensor).view(-1, size_0, size_1)
    num_pos = 0
    for i in range(len(label_train)):
        if label_train[i] == 1:
            num_pos = num_pos + 1
    print(num_pos / len(label_train))

    return X_train, label_train, X_valid, label_valid, X_test, label_test, X, labels


class Sample_model(nn.Module):
    def __init__(self, d_model, dim_out, dropout=0.1):
        super(Sample_model, self).__init__()
        self.linear_1 = nn.Linear(d_model, 1024)
        self.linear_2 = nn.Linear(1024, 128)
        self.linear_3 = nn.Linear(128, dim_out)
    def forward(self, x, mask = None):
        x = torch.sum(x, dim = 1)
        result = torch.relu(self.linear_1(x))
        result = torch.relu(self.linear_2(result))
        result = torch.relu(self.linear_3(result))
        result = torch.sigmoid(result)
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
    result = []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()  # move to device, e.g. GPU
            y = y.cuda()
            total = model(x, None)
            #preds = (F.sigmoid(total) > 0.5).squeeze(1)
            preds =(F.softmax(total, dim=1)[:, 0] > 0.5)
            result.extend(list(np.array(F.sigmoid(total).cpu())))
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return result

def train(model, loader_train, loader_val, iter_time = 100, use_gpu = True):
    print('>>>> training...')

    loss_function = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6)

    if (use_gpu):
        model = model.cuda()
        #loss_function = loss_function.cuda()

    examples_scores_pre = torch.zeros(TRAIN_NUM, 2)
    for _ in range(iter_time):
        model.train()
        for t, (x, y_id) in enumerate(loader_train):
            y = y_id[:,0]
            if (use_gpu):
                x = x.cuda().detach()
                y = y.unsqueeze(1).cuda().detach()
            y_pred = model(x, None)
            #pred_sigmod = F.sigmoid(y_pred)
            pred_softmax = F.softmax(y_pred, dim = 1)[:,0].unsqueeze(1)
            '''
            print(pred_softmax)
            weight = torch.ones(y.size(0))
            for i in range(y.size(0)):
                if y[i] == 0:
                    weight[i] = math.exp(2 * pred_softmax[i,0])
            weight = weight.unsqueeze(1).cuda()
            loss = loss_function(y_pred, y, weight=weight)
            '''
            loss = loss_function(pred_softmax, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 500 == 0:
                acc = check_accuracy(loader_val, model)
        '''
                if acc < acc_pre:
                    if down_num >= 5:
                        flag_worse = 1
                        break
                    else:
                        down_num = down_num + 1
                else:
                    down_num = 0
                    acc_pre = acc
        if flag_worse == 1:
            break
        '''
    torch.save(model.state_dict(), Model_Path)
    return True

def train_Focal(model, loader_train, loader_val, iter_time = 100, use_gpu = True):
    print('>>>> training...')

    loss_function = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6)

    if (use_gpu):
        model = model.cuda()
        #loss_function = loss_function.cuda()

    examples_scores_pre = torch.zeros(TRAIN_NUM, 2)
    for _ in range(iter_time):
        model.train()
        for t, (x, y_id) in enumerate(loader_train):
            y = y_id[:,0]
            if (use_gpu):
                x = x.cuda().detach()
                y = y.unsqueeze(1).cuda().detach()
            y_pred = model(x, None)
            pred_softmax = F.softmax(y_pred, dim = 1)[:,0].unsqueeze(1)

            weight = torch.ones(y.size(0))
            for i in range(y.size(0)):
                if y[i] == 1:
                    weight[i] = math.exp((1 - pred_softmax[i, 0]))
                elif y[i] == 0:
                    weight[i] = math.exp(pred_softmax[i, 0])
            weight = weight.unsqueeze(1).cuda()
            loss = loss_function(pred_softmax, y, weight=weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 500 == 0:
                acc = check_accuracy(loader_val, model)
        '''
                if acc < acc_pre:
                    if down_num >= 5:
                        flag_worse = 1
                        break
                    else:
                        down_num = down_num + 1
                else:
                    down_num = 0
                    acc_pre = acc
        if flag_worse == 1:
            break
        '''
    torch.save(model.state_dict(), Model_Path)
    return True

def train_PSKD(model, loader_train, loader_val, iter_time = 100, use_gpu = True, lr_decay_schedule = [20,80], Tem = 1):
    print('>>>> training...')

    loss_function = F.binary_cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_schedule, gamma=0.9)

    if (use_gpu):
        model = model.cuda()

    examples_scores_pre = torch.zeros(TRAIN_NUM, 2).cuda()
    for epoch in range(iter_time):
        alpha_t = 0.8 * ((epoch + 1) / iter_time)
        alpha_t = max(0, alpha_t)
        model.train()
        for t, (x, y_id) in enumerate(loader_train):
            y = y_id[:,0]

            example_id = y_id[:,1].long()
            if (use_gpu):
                x = x.cuda().detach()
                y = y.unsqueeze(1).cuda().detach()
            y_pred = model(x, None)
            y_pred_withtem = y_pred/Tem
            pred_softmax = F.softmax(y_pred_withtem, dim=1)[:,0].unsqueeze(1)
            if t > 100:
                soft_y = ((1 - alpha_t) * y) + (alpha_t * examples_scores_pre[example_id,0].unsqueeze(1)).detach()
            else:
                soft_y = y

            weight = torch.ones(y.size(0))
            for i in range(y.size(0)):
                if y[i] == 1 and pred_softmax[i,0] < 0.2:
                    weight[i] = math.exp((1-pred_softmax[i,0]))
                elif y[i] == 0 and pred_softmax[i,0] > 0.8:
                    weight[i] = math.exp(pred_softmax[i,0])

            weight = weight.unsqueeze(1).cuda()
            #print(pred_softmax.size(),soft_y.size(),weight.size())
            loss = loss_function(pred_softmax, soft_y, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            examples_scores_pre[example_id] = pred_softmax
            if t % 500 == 0:
                acc = check_accuracy(loader_val, model)

    torch.save(model.state_dict(), Model_Path)
    return True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    batch_size = 100
    train_x, train_y, valid_x, valid_y, test_x, test_y, all_x, all_y = load_data()
    example_num = train_x.size(0)
    example_id = torch.arange(0, example_num, 1)
    train_y = train_y.unsqueeze(-1)
    example_id = example_id.unsqueeze(-1)
    train_y = torch.cat((train_y,example_id),1)
    TRAIN_NUM = train_x.size(0)
    data_train = Data.TensorDataset(train_x, train_y)
    print(data_train[0][1])

    data_val = Data.TensorDataset(valid_x, valid_y)
    data_test = Data.TensorDataset(test_x, test_y)
    data_all = Data.TensorDataset(all_x, all_y)
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
    batch_all = all_y.size(0)
    loader_all = Data.DataLoader(
        dataset=data_all,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    transformer_half = make_model(d_out=2)
    train_Focal(transformer_half, loader_train, loader_val, iter_time = 500, use_gpu = True)
    #train_PSKD(transformer_half, loader_train, loader_val, iter_time = 500,lr_decay_schedule = [200,300,350,400,450],Tem = 2)
    check_accuracy(loader_test, transformer_half, False)
    result = check_accuracy(loader_all, transformer_half, False)
    df_result = pd.DataFrame(np.array(result).reshape(-1, 1))
    df_result.to_csv('possibility_attention_imdb.csv')


