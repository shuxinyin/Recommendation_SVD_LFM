import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def svd_read_data(file_path='../data/movie_lens_1m/u.csv'):
    import sys
    df = pd.read_csv(file_path, header=0, names=['user', 'item', 'rating', 'time'], sep=',', encoding='utf8')
    print(df.head())

    rate_mean = np.mean(df.rating.to_list())

    # total_nodes = set(list(df.user.unique()) + list(df.item.unique()))
    # map_dict = dict(zip(total_nodes, range(len(total_nodes))))
    user_nodes = list(df.user.unique())
    user_map_dict = dict(zip(user_nodes, range(len(user_nodes))))
    item_nodes = list(df.item.unique())
    item_map_dict = dict(zip(item_nodes, range(len(item_nodes))))
    n_user, n_item = len(user_nodes), len(item_nodes)

    # map user and item to number
    df['user'] = df.user.map(user_map_dict)
    df['item'] = df.item.map(item_map_dict)
    user_item_dic = df.groupby('user')['item'].apply(list).to_dict()  # get {user:item_list} dict

    df_count = pd.DataFrame(df.groupby('user').size())
    df_1w = df_count.sort_values([0], ascending=False)[:200]

    top_1w_list = list(df_1w.reset_index().user.unique())
    df_test = df[df.user.isin(top_1w_list)]
    df_test = df_test.groupby('user').apply(pd.DataFrame.sample, n=1)

    df_test.index = df_test.user * n_item + df_test.item
    df.index = df.user * n_item + df.item
    df_train = df.drop(df_test.index)

    print(df_train.shape, df_test.shape)

    return df_train, df_test, n_user, n_item, rate_mean, user_item_dic


def reproduce_svd_data_read(file_path='../data/movie_lens_1m/u.csv', test_size=0.1):
    df = pd.read_csv(file_path, header=0, names=['user', 'item', 'rating', 'time'], sep=',', encoding='utf8')
    print(df.shape)

    n_user = df.user.unique().shape[0]
    n_item = df.item.unique().shape[0]
    rate_mean = np.mean(df.rating.to_list())

    # total_nodes = set(list(df.user.unique()) + list(df.item.unique()))
    # map_dict = dict(zip(total_nodes, range(len(total_nodes))))
    user_nodes = list(df.user.unique())
    user_map_dict = dict(zip(user_nodes, range(len(user_nodes))))
    item_nodes = list(df.user.unique())
    item_map_dict = dict(zip(item_nodes, range(len(item_nodes))))

    # map user and item to number
    df['user'] = df.user.map(user_map_dict)
    df['item'] = df.item.map(item_map_dict)

    train_data, test_data = train_test_split(df, test_size=test_size)
    print(rate_mean, train_data.shape, test_data.shape)  # 3.501746517983075
    return train_data, test_data, n_user, n_item, rate_mean


class MyCollator(object):
    def __init__(self, all_item_num, user_item_dict, negative_ratio=0, train_bool='train'):
        self.all_item_num = all_item_num
        self.user_item_dict = user_item_dict
        self.negative_ratio = negative_ratio
        # self.train_bool = train_bool

    def __call__(self, batch):
        # do something with batch and self.params
        user_mini_batch, item_mini_batch, rate_mini_batch = list(), list(), list()
        user, item, rate = batch[0][0], batch[0][1], batch[0][2]
        for k in range(self.negative_ratio):
            user_mini_batch.append(user)
            true_item_list = self.user_item_dict[user]
            if k < 1:
                item_mini_batch.append(item)
                rate_mini_batch.append(rate)
            else:
                rate_mini_batch.append(0)
                while True:
                    item_neg = random.choice(range(self.all_item_num))
                    if item_neg not in true_item_list:
                        item_mini_batch.append(item_neg)
                        break

        user_mini_batch = torch.from_numpy(np.array(user_mini_batch))
        item_mini_batch = torch.from_numpy(np.array(item_mini_batch))
        rate_mini_batch = torch.from_numpy(np.array(rate_mini_batch))
        return [user_mini_batch, item_mini_batch, rate_mini_batch]


class svd_dataset(Dataset):
    def __init__(self, df, all_item_num, user_item_dict, negative_ratio=1, train_bool=True):
        self.user_list = df.user.to_list()
        self.item_list = df.item.to_list()
        self.rating_list = df.rating.to_list()

        self.all_item_num = all_item_num
        self.user_item_dict = user_item_dict
        self.negative_ratio = negative_ratio

        self.train_bool = train_bool

    def __len__(self):
        return len(self.user_list) * self.negative_ratio


    def __getitem__(self, index):
        if index < len(self.user_list):
            user = self.user_list[index]
            item = self.item_list[index]
            rating = self.rating_list[index]
        else:
            index = index % len(self.user_list)
            user = self.user_list[index]
            true_item_list = self.user_item_dict[user]
            while True:
                item_neg = random.choice(range(self.all_item_num))
                if item_neg not in true_item_list:
                    item = item_neg
                    rating = 0
                    break
        return user, item, rating


if __name__ == "__main__":
    data_path = '../../data/movie_lens_1m/ratings.csv'
    df_train, df_test, n_user, n_item, rate_mean, user_item_dic = svd_read_data(file_path=data_path)

    train_dataset = svd_dataset(df_train, n_item, user_item_dic)
    test_dataset = svd_dataset(df_test, n_item, user_item_dic)

    my_collator = MyCollator(n_item, user_item_dic, negative_ratio=100)
    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4,
                                   collate_fn=my_collator)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                  collate_fn=my_collator)  # batch size must be one
    for i, batch in enumerate(train_data_loader):
        assert len(batch[0]) == len(batch[1]) == len(batch[2])
        print('--', i, len(batch), len(batch[0]))
        break
