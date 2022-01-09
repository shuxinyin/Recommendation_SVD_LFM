import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

torch.cuda.current_device()

from utils import load_config
from model import SVD
from evaluation import mse_rmse_mae, metric
from dataloader import reproduce_svd_data_read, svd_read_data, svd_dataset, MyCollator


def mse_loss_fn(pred, rate, rate_max=5.0):
    # c_ui = 1+  0.2 * torch.abs(rate - (rate_max)/2)
    mse = torch.pow(rate - pred, 2)
    return torch.mean(mse)


def main(config):
    print(torch.cuda.device_count(), torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # device = torch.device(config.gpu)
    cudnn.benchmark = True
    #  data loader
    df_train, df_test, n_user, n_item, rate_mean, user_item_dic = svd_read_data(
        file_path=config.data_path)

    train_dataset = svd_dataset(df_train, n_item, user_item_dic, negative_ratio=config.negative_ratio)
    test_dataset = svd_dataset(df_test, n_item, user_item_dic, negative_ratio=1)

    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    collator_for_test = MyCollator(n_item, user_item_dic, negative_ratio=300)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                  collate_fn=collator_for_test)  # batch size must be one

    #  model definition  1. model define 2. optimizer define 3.loss func define
    model = SVD(n_user, n_item, config.emb_dim, rate_mean=rate_mean, rate_min=0, rate_max=5)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=0.01, amsgrad=False)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=float(config.lr), weight_decay=1e-5)
    writer = SummaryWriter()  # for visualization

    # train
    count = 0
    for epoch in range(config.epochs):
        model.train()
        loss_total = list()
        start_time = time.time()
        for i, (user, item, rate) in enumerate(train_data_loader):
            user = user.cuda().long()
            item = item.cuda().long()
            rate = rate.cuda()

            model.zero_grad()
            predict = model.forward(user, item)
            loss = mse_loss_fn(predict, rate)
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/loss', loss.item(), count)
            loss_total.append(loss.item())
            count += 1

        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

        #  evaluation
        if epoch % 2 == 0:
            model.eval()
            # mse, rmse, mae = mse_rmse_mae(model, test_data_loader)
            # print("Epoch: %03d; MSE = %.4f, RMSE = %.4f, MAE = %.4f" % (epoch, mse, rmse, mae))
            hr, ndcg = metric(model, test_data_loader, top_k=10)
            print("Epoch: %03d; HR = %.4f, NDCG = %.4f, time=%.4f" % (epoch, hr, ndcg, time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='svd')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(type(config.lr), config.lr, float(config.lr))

    main(config)
