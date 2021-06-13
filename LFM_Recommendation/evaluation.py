import torch
import math
import numpy as np


def hit_ratio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def ndcg(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def get_TopK(pred, item, top_k=1):
    dic = dict(zip(item, pred))
    sort_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    top_k = list(sort_dic.keys())[:top_k]
    print(top_k)


def metric(model, test_data_loader, top_k=1):
    hr_list, ndcg_list = list(), list()
    for i, (user, item, rate) in enumerate(test_data_loader):
        user = user.cuda().long()
        item = item.cuda().long()
        gt_item = item[0].item()

        predict = model.forward(user, item)
        sores, indices = torch.topk(predict, k=top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        # print("---", gt_item, recommends)
        hr_list.append(hit_ratio(recommends, gt_item))
        ndcg_list.append(ndcg(recommends, gt_item))

    return np.mean(hr_list), np.mean(ndcg_list)


def mse_rmse_mae(model, test_data_loader):
    mse_list, rmse_list, mae_list = [], [], []
    for i, (user, item, rate) in enumerate(test_data_loader):
        user = user.cuda().long()
        item = item.cuda().long()
        rate = rate.cuda()

        predict = model.forward(user, item)

        mse = torch.mean(torch.pow(predict - rate, 2.0))
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(predict - rate))

        mse_list.append(mse.item())
        rmse_list.append(rmse.item())
        mae_list.append(mae.item())

    return np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list)


if __name__ == "__main__":
    pre = [0.1, 0.2, 0.3]
    item = ['a', 'b', 'c']
    get_TopK(pre, item)
