import os
import random

import pandas as pd
import numpy as np


# Get list whose item count > 2
def sample(df_tmp, n=500):
    # random.set_seed(123)

    # pos dict and pos list
    pos_dict = dict(df_tmp.groupby('user_id')['item_id'].apply(list))
    pos_list = []
    for key in pos_dict.keys():
        pos_list.extend(pos_dict[key])
    pos_list = list(np.unique(pos_list))
    # sample list
    item_count = pd.DataFrame(df_tmp.reset_index().item_id.value_counts()).reset_index()
    item_count.columns = ['item_id', 'item_count']
    item_count_list = list(item_count[item_count.item_count >= 3].item_id.unique())  # item count >2 list
    print(len(item_count_list))
    # Positive Sample
    # print(df_tmp.item_id.isin(item_count_list).astype(int))
    df_pos = df_tmp.sample(n, weights=df_tmp.item_id.isin(item_count_list).astype(int),
                           random_state=1)  # sample those item count >2 list
    df = df_tmp.drop(df_pos.index)[['ipc_bool', 'user_id', 'item_id']]
    df_pos['true_false_edge'] = 1
    df_pos = df_pos[['ipc_bool', 'user_id', 'item_id', 'true_false_edge']]
    # Negtive Sample
    false_edges = []
    for row_index, row in df_pos.iterrows():
        while True:
            neg_dst = random.choice(pos_list)
            if neg_dst not in pos_dict[row.user_id]:
                false_edges.append([row.ipc_bool, row.user_id, neg_dst, 0])
                break

    df_neg = pd.DataFrame(false_edges, columns=['ipc_bool', 'user_id', 'item_id', 'true_false_edge'])
    print("sample train and valid shape", df.shape, pd.concat([df_pos, df_neg]).shape)
    return df.reset_index(drop=True), pd.concat([df_pos, df_neg]).reset_index(drop=True)


def test():
    dic = {"abc": ["bcd", "vbc", "dka"],
           "bcd": ["jkl", "sda", "das"]}
    df = pd.DataFrame.from_dict(dic, orient="index").reset_index()
    print(df)

    df_list = []
    for i in range(3):
        print(df.loc[:, ['index', i]])
        df_tmp = df.loc[:, ['index', i]]
        df_tmp.columns = ["com1", "com2"]
        df_list.append(df_tmp)
    df = pd.concat(df_list, axis=0)
    print(df)


if __name__ == "__main__":
    # file_path = '../data/movie_lens_1m/ratings.csv'
    # df = pd.read_csv(file_path, header=0, names=['user_id', 'item_id', 'rating', 'time'], sep=',', encoding='utf8')
    # df['ipc_bool'] = 1
    # print(len(list(df.item_id.unique())))
    # train, valid = sample(df, n=7000)
    # item_valid = list(valid.item_id.unique())
    # item_train = list(train.item_id.unique())
    # none_item = [i for i in item_valid if i not in item_train]
    # print(none_item)

    test()
