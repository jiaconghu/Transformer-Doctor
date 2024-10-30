import os

import cv2
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

matplotlib.use('AGG')


def heatmap(vals, fig_path, fig_w=None, fig_h=None, annot=False):
    if fig_w is None:
        fig_w = vals.shape[1]
    if fig_h is None:
        fig_h = vals.shape[0]

    f, ax = plt.subplots(figsize=(fig_w, fig_h), ncols=1)
    sns.heatmap(vals, ax=ax, annot=annot)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def draw_group(data, fig_path):
    # features = features[:, :, 0:50]

    t = data.shape[0]
    s = data.shape[1]
    d = data.shape[2]

    x = np.tile(np.arange(0, d), t * s)
    y = data.flatten()
    z = np.repeat(np.arange(0, t), s * d)

    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=x, y=y, hue=z, s=20, alpha=0.5, style=z, palette=sns.color_palette("hls", t))
    # sns.lineplot(x="channel", y="value", data=data_plot, hue=z)

    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def draw_single(data, fig_path):
    np.set_printoptions(threshold=np.inf)
    n = data.shape[0]  # 一个坐标几种数据
    # s = data.shape[1]
    d = data.shape[1]  # 坐标长度

    x = np.tile(np.arange(0, d), n)  # [0-d,0-d,...,0-d] * 外n
    # print(x)
    y = data.flatten()
    z = np.repeat(np.arange(0, n), d)  # [0-0,1-1,...,n-n] * 内d
    # print(z)

    plt.figure(figsize=(15, 10))
    # sns.lineplot(x=x, y=y, hue=z, style=z, palette=sns.color_palette("hls", n))
    sns.scatterplot(x=x, y=y, hue=z, style=z, palette=sns.color_palette("hls", n))

    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def draw_box(data, fig_path):
    C, N, D = data.shape

    x = np.tile(np.arange(0, D), N * C)
    z = np.repeat(np.arange(0, C), N * D)
    y = data.flatten()

    plt.figure(figsize=(15, 10))
    sns.boxplot(x=x, y=y, hue=z, palette=sns.color_palette("hls", C))
    # sns.scatterplot(x=x, y=y, hue=z, s=20, alpha=0.5, style=z, palette=sns.color_palette("hls", C))
    # plt.show()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


# def feature_distribute(features, fig_path):
#     plt.figure(figsize=(15, 10))
#
#     s = features.shape[0]
#     d = features.shape[1]
#
#     x = np.tile(np.arange(0, d), s)
#     y = features.flatten()
#
#     sns.scatterplot(x=x, y=y, s=20, alpha=0.5)
#     # sns.lineplot(x=x, y=y)
#
#     plt.savefig(fig_path, bbox_inches='tight')
#     plt.clf()


def imshow(img, title, fig_path):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True, nrow=10)
    npimg = img.numpy()
    # fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.title(title)
    # plt.show()

    plt.title(title)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.clf()


def save_img_by_cv2(img, path):
    img_dir, _ = os.path.split(path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
