import numpy as np
import pandas as pd
import os
import sys
#import click
import random
import collections
#from gensim.models import Word2Vec, KeyedVectors
#from nltk.tokenize import WhitespaceTokenizer
import random


movie_path = 'dataset_imdb/IMDb movies.csv'
rating_path = 'dataset_imdb/IMDb rating.csv'

def save_list(file_name, list_save, d):
    file_path = 'lists_dicts/'
    fileObject = open(file_name, 'w',encoding="utf-8")
    if d == 1:
        for li in list_save:
            fileObject.write(str(li))
            fileObject.write('\n')
    elif d == 2 or d == 3:
        for li in list_save:
            for i in range(len(li)):
                fileObject.write(str(li[i]))
                if i != len(li) - 1:
                    fileObject.write(",")
            fileObject.write('\n')
    fileObject.close()

def value_to_category(value, standrads):
    if value <= standrads[0]:
        return 0
    elif value > standrads[0] and value <= standrads[1]:
        return 1
    elif value > standrads[1] and value <= standrads[2]:
        return 2
    elif value > standrads[2] and value <= standrads[3]:
        return 3
    elif value > standrads[3] and value <= standrads[4]:
        return 4
    elif value > standrads[4] and value <= standrads[5]:
        return 5
    elif value > standrads[5] and value <= standrads[6]:
        return 6
    elif value > standrads[6] and value <= standrads[7]:
        return 7
    else:
        return 8

def delete_useless():
    f_movie = open(movie_path, errors='ignore')
    df_movie = pd.read_csv(f_movie)
    #df_movie = df_movie.drop(df_movie[(df_movie.avg_vote < 6.0) & (df_movie.avg_vote > 5.0)].index)
    #df_movie_eng = df_movie.loc[(df_movie['language'] == 'English') | (df_movie['language'] == 'None')]
    df_movie_data = df_movie.drop(df_movie.columns[[1, 2, 3, 8, 13, 16, 17, 18, 19]], axis=1 )
    #df_movie_label = df_movie_eng.iloc[:,12]
    #df_movie_data = df_movie.drop(df_movie.columns[[12]], axis=1)
    #movie_label = np.array(df_movie_label)
    data_isnull = df_movie_data.isnull()
    for i in range(len(data_isnull)):
        for j in range(1,13):
            if data_isnull.iloc[i,j] == True:
                if j in [3, 9, 10, 11, 12]:
                    df_movie_data.iloc[i,j] = 0
                elif j in [2, 4, 5, 6, 7, 8]:
                    df_movie_data.iloc[i,j] = 'unknown'
    divides_lists = []
    divides_standrads = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99]
    number_features = [3, 9, 10, 11, 12]
    for j in number_features:
        divides_list = []
        feature_value_list = list(df_movie_data.iloc[:, j])
        feature_list_sorted = sorted(feature_value_list, reverse=False)
        for ds in divides_standrads:
            divides_list.append(feature_list_sorted[int(ds * len(feature_list_sorted))])
        divides_lists.append(divides_list)
        print(divides_list)
    for i in range(len(df_movie_data)):
        if 'TV Movie' in df_movie_data.iloc[i,1]:
            idx_1 = df_movie_data.iloc[i, 1].find('20')
            idx_2 = df_movie_data.iloc[i, 1].find('19')
            idx_3 = df_movie_data.iloc[i, 1].find('18')
            if idx_1 != -1:
                df_movie_data.iloc[i, 1] = df_movie_data.iloc[i, 1][idx_1:idx_1 + 4]
            elif idx_2 != -1:
                df_movie_data.iloc[i, 1] = df_movie_data.iloc[i, 1][idx_2:idx_2 + 4]
            elif idx_3 != -1:
                df_movie_data.iloc[i, 1] = df_movie_data.iloc[i, 1][idx_3:idx_3 + 4]
        for j in range(len(number_features)):
            #max_value = df_movie_data.iloc[:, j].max()
            #unit_value = round(max_value/10, 1)
            df_movie_data.iloc[i, number_features[j]] = \
                value_to_category(df_movie_data.iloc[i, number_features[j]], divides_lists[j])

    df_movie_data.to_csv('filtered_data.csv')
    movie_data = np.array(df_movie_data)
    print(np.shape(movie_data))
    return movie_data

def dict_generate(movie_data, cutoff = 3, categorial_features = [2, 4, 5, 6, 7, 8]):
    #movie_data = delete_useless()
    dicts = []
    for i in range(len(categorial_features)):
        dicts.append(collections.defaultdict(int))
    for movie in movie_data:
        for i in range(len(categorial_features)):
            movie_feature_splited = movie[categorial_features[i]].split(',')
            for feature in movie_feature_splited:
                feature = feature.strip()
                dicts[i][feature] += 1
    #print(dicts[0])
    for i in range(len(categorial_features)):
        if i not in [1,2]:
            dicts[i] = filter(lambda x: x[1] >= cutoff, dicts[i].items())
        else:
            dicts[i] = filter(lambda x: x[1] >= 0, dicts[i].items())
        dicts[i] = sorted(dicts[i], key=lambda x: (-x[1], x[0]))
        vocabs, _ = list(zip(*dicts[i]))
        dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
        dicts[i]['<unk>'] = 0
        #print(dicts[i])
        pd.DataFrame([dicts[i]]).to_csv('dicts/'+str(i)+'.csv')

    return dicts

def preprocess(number_features = [3, 9, 10, 11, 12], categorial_features = [2, 4, 5, 6, 7, 8]):
    '''lists'''
    date_year = []
    date_month = []
    date_day = []
    genre_list = []
    duration_list = []
    country_list = []
    director_list = []
    writer_list = []
    company_list = []
    actor_list = []
    average_vote_list = []
    vote_list = []
    review_list = []
    critic_list = []

    movie_data = delete_useless()
    categorial_dicts = dict_generate(movie_data)
    #keyed_vectors = KeyedVectors.load('embedding_200.kv')
    #word_unk_vector = list(np.zeros(200))
    for movie in movie_data:
        '''
        for i in range(len(string_features)):
            string_splited = WhitespaceTokenizer().tokenize(movie[string_features[i]])
            word_list = []
            for word in string_splited:
                try:
                    word_vector = list(keyed_vectors[word])
                    word_list.append(word_vector)
                except:
                    word_list.append(word_unk_vector)
            if i == 0:
                title_list.append(word_list)
            elif i == 1:
                o_title_list.append(word_list)
            else:
                plot_descrption.append(word_list)
        '''
        date_splited = movie[1].split('-')
        date_year.append(int(date_splited[0])-1894)
        if len(date_splited) >= 2:
            date_month.append(int(date_splited[1])-1)
        else:
            date_month.append(random.randint(0, 11))
        if len(date_splited) >= 3:
            date_day.append(int(date_splited[2])-1)
        else:
            date_day.append(random.randint(0, 30))

        for i in range(len(number_features)):
            feature_value = movie[number_features[i]]
            if i == 0:
                duration_list.append(int(feature_value))
            elif i == 1:
                average_vote_list.append(int(feature_value))
            elif i == 2:
                vote_list.append(int(feature_value))
            elif i == 3:
                review_list.append(int(feature_value))
            else:
                critic_list.append(int(feature_value))

        for i in range(len(categorial_features)):
            res_list = []
            movie_feature_splited = movie[categorial_features[i]].split(',')
            #print(movie_feature_splited)
            for feature_key in movie_feature_splited:
                feature_key = feature_key.strip()
                if feature_key not in categorial_dicts[i]:
                    res = categorial_dicts[i]['<unk>']
                else:
                    res = categorial_dicts[i][feature_key]
                res_list.append(res)
            if i == 0:
                genre_list.append(res_list)
            elif i == 1:
                country_list.append(res_list)
            elif i == 2:
                director_list.append(res_list)
            elif i == 3:
                writer_list.append(res_list)
            elif i == 4:
                company_list.append(res_list)
            else:
                actor_list.append(res_list)
    print("preprocess over!!!")
    print(duration_list)
    print(average_vote_list)
    print(vote_list)
    print(review_list)
    print("critic",critic_list)
    print("date_year",date_year)
    print(type(critic_list[0]))
    print(len(duration_list), len(average_vote_list), len(vote_list), len(review_list), len(critic_list))

    dict_sizes = [len(categorial_dicts[idx]) for idx in range(0, len(categorial_dicts))]
    with open(os.path.join('lists_dicts/', 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [9] * 5 + [128, 12, 31] + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))
    save_list('date_year.txt', critic_list, 1)
    save_list('date_month.txt', date_month, 1)
    save_list('date_day.txt', date_day, 1)
    save_list('genre.txt', genre_list, 2)
    save_list('duration.txt', duration_list, 1)
    save_list('country.txt', country_list, 2)
    save_list('director.txt', director_list, 2)
    save_list('writer.txt', writer_list, 2)
    save_list('company.txt', company_list, 2)
    save_list('actor.txt', actor_list, 2)
    save_list('average_vote.txt', average_vote_list, 1)
    save_list('vote.txt', vote_list, 1)
    save_list('review.txt', review_list, 1)
    save_list('critic.txt', date_year, 1)
    print("save over!!!")

if __name__ == '__main__':
    #preprocess()
    for j in range(10):
        save_list('critic'+str(j)+'.txt', [int(i) for i in range(200000)], 1)
    #delete_useless()
    #dict_generate()
