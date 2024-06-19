"""
线性回归的简洁实现,使用nn模块
"""

import numpy as np
import torch
from torch.utils import data #引入处理数据的模块
from torch import nn  #nn是神经网络的缩写
from d2l import torch as d2l


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
'''生成数据'''
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    """调用框架中现有的API来读取数据

    Args:
        data_arrays (tensor): features, labels
        batch_size (tensor): 小梯度，hyperparameters
        is_train (bool, optional): 是否希望在每个迭代周期打乱数据. Defaults to True.

    Returns:
        _type_: 喂数据
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
'''与上个文件中定义的data_iter函数类似'''
data_iter = load_array((features, labels), batch_size)
'''使用iter构造python迭代器，使用next从迭代器中获取第一项'''
print(next(iter(data_iter)))
'''第一参数指定输入特征形状：2
第二个指定输出特征形状：1'''
net = nn.Sequential(nn.Linear(2, 1)) #网络添加参数

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
'''均方误差'''
loss = nn.MSELoss()
'''SGD随机梯度下降'''
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        '''进行模型更新'''
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)