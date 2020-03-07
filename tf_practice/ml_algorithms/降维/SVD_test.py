#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    image_sm = load_digits()

    print(image_sm.images[0])
    plt.gray()
    plt.matshow(image_sm.images[1])
    plt.show()

    svd = TruncatedSVD()

    data = image_sm.images
    label = image_sm.target
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2)