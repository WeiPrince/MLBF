import pandas as pd
import numpy as np
import random
import csv

"""
生成负例
dataset：pandas读取到的数据
size：生成的条数
"""


def getlist(data):
    res = set()
    for x in data:
        res.update(x.split(","))
    res = list(res)
    return res


def create_negative(dataset, size):
    negative_data = []
    genre_list = getlist(dataset['genre'])
    country_list = getlist(dataset['country'])
    director_list = getlist(dataset['director'])
    writer_list = getlist(dataset['writer'])
    production_company_list = getlist(dataset['production_company'])
    actor_list = getlist(dataset['actors'])
    for i in range(size):
        # date_published：格式2021/5/7
        date_published = str(random.randint(1922, 2021)) + "/" + str(random.randint(1, 12)) + "/" + str(
            random.randint(1, 31))
        # genre
        genre = ",".join(random.sample(genre_list, random.randint(1, 4)))
        # duration
        duration = str(random.randint(0, 8))
        # country
        country = random.choice(country_list)
        # director
        director = ",".join(random.sample(director_list, random.randint(1, 5)))
        # writer
        writer = ",".join(random.sample(director_list, random.randint(1, 3)))
        # production_company
        production_company = random.choice(production_company_list)
        # actors
        actors = ",".join(random.sample(actor_list, random.randint(1, 15)))
        # avg_vote
        avg_vote = str(random.randint(0, 8))
        # votes
        votes = str(random.randint(0, 8))
        # reviews_from_users
        reviews_from_users = str(random.randint(0, 8))
        # reviews_from_critics
        reviews_from_critics = str(random.randint(0, 8))
        negative_data.append(
            [date_published, genre, duration, country, director, writer, production_company, actors, avg_vote, votes,
             reviews_from_users, reviews_from_critics])
    return negative_data


if __name__ == '__main__':
    dataset = pd.read_csv('../file/file.csv', encoding='ISO-8859-1')
    df = pd.DataFrame(np.array(create_negative(dataset, 80000)))
    df.to_csv("neg.csv")
    print("生成完成")