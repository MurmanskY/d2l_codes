'''
手动实现线性回归模型
'''
import torch
import random
import matplotlib
from d2l import torch as d2l  # 之前实现的函数包


def synthetic_data(w, b, num_examples):
    '''生成y=Xw+b+噪声'''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 注意matmul()函数的用法
    y += torch.normal(0, 0.1, size=y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  # 获取所有下标
    random.shuffle(indices)  # 对下标随机排序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]  # 迭代作用，一次喂一小部分，直到全部喂完

def linreg(X, w, b):
    '''线性回归模型'''
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sqd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad(): #在上下文管理器内部，禁用自动梯度计算，提高性能，节省内存
        for param in params: #遍历模型所有参数，
            param -= lr * param.grad / batch_size # 使用平均梯度更新参数
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 生成数据用的w和b，仍然是添加了一个噪声的
# 最后的预测结果应该会和true_w和true_b相近

batch_size = 10

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.003
num_epochs = 200
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sqd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - b}')
print(w.detach(), '\n', b.detach())


