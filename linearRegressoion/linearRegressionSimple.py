"""
线性回归的简洁实现,使用nn模块
"""

import numpy as np
import torch
from torch.utils import data #引入处理数据的模块
from d2l import torch as d2l


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)

features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    """
    构造一个PyTorch的数据迭代器
    :param data_arrays:
    :param batch_size:
    :param is_train: 是否希望数据迭代器在每个迭代周期内打乱数据
    :return:
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array(features, labels)