import numpy as np
import pandas as pd
import os
import random
import collections
import random

pd.set_option('display.max_columns', 20)
movie_path = 'dataset_imdb/filtered_data_fake.csv'

def save_list(file_name, list_save, d, file_path='lists_dicts/'):
    fileObject = open(file_path + file_name, 'w', encoding='gbk')
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
    df_movie_data = pd.read_csv(f_movie)
    df_movie_label = df_movie_data.iloc[:, 11]
    df_movie_data = df_movie_data.drop(df_movie_data.columns[[11]], axis=1)
    movie_label = np.array(df_movie_label).reshape(-1)
    data_isnull = df_movie_data.isnull()
    for i in range(len(data_isnull)):
        for j in range(1, 10):
            if data_isnull.iloc[i, j] == True:
                if j in [2, 8, 9, 10]:
                    df_movie_data.iloc[i, j] = random.randint(1, 90)
                elif j in [1, 3, 4, 5, 6, 7]:
                    df_movie_data.iloc[i, j] = 'unknown'

    #df_movie_data.to_csv('filtered_data_original.csv')
    movie_data = np.array(df_movie_data)
    print(movie_label)
    print(np.shape(movie_data))
    return movie_data, movie_label


def dict_generate(movie_data, cutoff=3, categorial_features=[1, 3, 4, 5, 6, 7]):
    dicts = []
    for i in range(len(categorial_features)):
        dicts.append(collections.defaultdict(int))
    for movie in movie_data:
        for i in range(len(categorial_features)):
            movie_feature_splited = movie[categorial_features[i]].split(',')
            for feature in movie_feature_splited:
                feature = feature.strip()
                dicts[i][feature] += 1
    # print(dicts[0])
    for i in range(len(categorial_features)):
        if i not in [1, 2]:
            dicts[i] = filter(lambda x: x[1] >= cutoff, dicts[i].items())
        else:
            dicts[i] = filter(lambda x: x[1] >= 0, dicts[i].items())
        dicts[i] = sorted(dicts[i], key=lambda x: (-x[1], x[0]))
        vocabs, _ = list(zip(*dicts[i]))
        dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
        dicts[i]['<unk>'] = 0
        # print(dicts[i])
        pd.DataFrame([dicts[i]]).to_csv('dicts/' + str(i) + '.csv')

    return dicts


def preprocess(number_features=[2, 8, 9, 10], categorial_features=[1, 3, 4, 5, 6, 7]):
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
    vote_list = []
    review_list = []
    critic_list = []

    movie_data, movie_label = delete_useless()
    categorial_dicts = dict_generate(movie_data)

    for movie in movie_data:
        if '/' in movie[0]:
            date_splited = movie[0].split('/')
        else:
            date_splited = movie[0].split('-')
        date_year.append(int(date_splited[0]) - 1894)
        date_month.append(int(date_splited[1]) - 1)
        date_day.append(int(date_splited[2]) - 1)

        for i in range(len(number_features)):
            feature_value = movie[number_features[i]]
            if i == 0:
                duration_list.append(int(feature_value))
            elif i == 1:
                vote_list.append(int(feature_value))
            elif i == 2:
                review_list.append(int(feature_value))
            else:
                critic_list.append(int(feature_value))

        for i in range(len(categorial_features)):
            res_list = []
            movie_feature_splited = movie[categorial_features[i]].split(',')
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
    movie_label = movie_label.reshape(-1, 1)
    np.savetxt('lists_dicts/label.txt', movie_label, fmt='%d')
    print("preprocess over!!!")

    dict_sizes = [len(categorial_dicts[idx]) + 1 for idx in range(0, len(categorial_dicts))]
    with open(os.path.join('lists_dicts/', 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [9] * 4 + [128, 12, 31] + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))
    save_list('date_year.txt', date_year, 1)
    save_list('date_month.txt', date_month, 1)
    save_list('date_day.txt', date_day, 1)
    save_list('genre.txt', genre_list, 2)
    save_list('duration.txt', duration_list, 1)
    save_list('country.txt', country_list, 2)
    save_list('director.txt', director_list, 2)
    save_list('writer.txt', writer_list, 2)
    save_list('company.txt', company_list, 2)
    save_list('actor.txt', actor_list, 2)
    # save_list('average_vote.txt', average_vote_list, 1)
    save_list('vote.txt', vote_list, 1)
    save_list('review.txt', review_list, 1)
    save_list('critic.txt', critic_list, 1)
    print("save over!!!")


if __name__ == '__main__':
    #delete_useless()
     preprocess()
