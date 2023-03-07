import sys
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from torchvision.transforms import transforms
import albumentations as A

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_single_image(image, grid=0, title=None, call=False):
    plt.grid(grid)  # 显示网格线
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    if not call:
        plt.show()


def plot_multi_images(*images, grid=0, title=None):
    n = len(images)
    assert n > 1, "Func needs at least 2 image as param!"
    row_colum = 100 + n * 10
    plt.figure(figsize=(5 * n, 8), dpi=60)
    if title is not None:
        plt.title(title)
    for i, image in enumerate(images):
        plt.subplot(row_colum + i + 1)
        plot_single_image(image, grid=grid, call=True)
    plt.show()


def plot_single_heatmap(array, cmap='jet', title=None, dpi=60):
    plt.figure(figsize=(5, 5), dpi=dpi)
    if title is not None:
        plt.title(title)
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    sns.heatmap(array, square=True, cmap=cmap)


def plot_multi_heatmaps(*arrays, cmap='jet', title=None, row=None, col=None):
    n = len(arrays)
    if row is not None:
        col = int(np.ceil(n / row))
    else:
        if col is not None:
            row = int(np.ceil(n / col))
        else:
            row = int(np.sqrt(n))
            col = int(np.ceil(n / row))

    fig, ax = plt.subplots(row, col, figsize=(6 * col, 4 * row), dpi=60)
    if title is not None: fig.suptitle(title)

    for i, array in enumerate(arrays):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        r = i // col
        c = i % col
        axi = ax[r, c] if row > 1 else ax[c]
        sns.heatmap(array, square=True, cmap=cmap, ax=axi)
    plt.show()


def img_mask(img, array: np.ndarray, alpha=1.0, beta=0.2, cmap="jet"):
    array = array.copy()
    array = (array - array.min()) / (array.max() - array.min()) ** alpha * 255.0

    cmap = plt.get_cmap(cmap)
    cmap = np.array([cmap(i) for i in range(cmap.N)]) * 255.0
    cmap = cmap.astype(np.uint8)[..., :3]
    cimg = np.zeros_like(img)
    for i in range(256):
        cimg[np.where(array.astype(np.uint8) == i)] = cmap[i]
    attention = cimg * (beta + (img.astype(np.float32) / 255.0) * (1 - beta))
    return cimg.astype(np.uint8), attention.astype(np.uint8)


def plot(y,
         x=None,
         func="plot",
         color=None,
         labels=None,
         xlim=None,
         size=None,
         grid=False,
         title=None,
         call=False, **kwargs):
    f = getattr(plt, func)
    if color is None:
        color = default_colors
    if size is not None:
        plt.figure(figsize=size)
    if xlim is not None:
        plt.xlim(xlim)
    if title is not None:
        plt.title(title)
    plt.grid(grid)
    if x is not None:
        if isinstance(y, list):
            for i, yi in enumerate(y):
                if labels is not None:
                    f(x, yi, color=color[i % len(color)], label=labels[i % len(labels)], **kwargs)
                else:
                    f(x, yi, color=color[i % len(color)], **kwargs)
        else:
            f(x, y, color[0])
    else:
        if isinstance(y, list):
            for i, yi in enumerate(y):
                if labels is not None:
                    f(yi, color=color[i % len(color)], label=labels[i % len(labels)], **kwargs)
                else:
                    f(yi, color=color[i % len(color)], **kwargs)
        else:
            f(y, color[0])
    if labels is not None:
        plt.legend()
    if not call:
        plt.show()
