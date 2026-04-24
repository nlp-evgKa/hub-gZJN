# 尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个4维向量，哪一维数字最大就属于第几类

"""
#定义一个TorchModel 的类  ，继承nn.Moudle基类。
##第二行是在初始化类  等于初始化具体实体

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)  # 线性层 输出四个数值
        self.activation = torch.softmax  # 涉及到概率分布，用这个
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    #训练阶段是有标签的 用于训练  没有标签的时候，就直接用作输出预测值
    # def forward(self, x, y=None):
    #     x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
    #     y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
    #     if y is not None:
    #         return self.loss(y_pred, y)  # 预测值和真实值计算损失
    #     else:
    #         return y_pred  # 输出预测结果
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, 4)
        if y is not None:
            # 训练模式：cross_entropy内部会做softmax，直接传入logits
            return self.loss(logits, y.squeeze())  # y需要变成一维
        else:
            # 预测模式：手动做softmax得到概率分布
            return torch.softmax(logits, dim=1)
        
# 生成4维一个样本, 样本的生成方法，代表了我们要学习的规律
# 哪一维数字最大就属于第几类
# def build_sample():
#     x = np.random.random(4)
#     if x[0] >= x[3] and x[0] >= x[1] and x[0] >= x[2] :
#         return x, 0
#     elif x[1] >= x[3] and x[1] >= x[0] and x[1] >= x[2]   :
#         return x, 1
#     elif x[2] >= x[3] and x[2] >= x[0] and x[2] >= x[1]   :
#         return x, 2
#     else:
#         return x, 3
def build_sample():
    x = np.random.random(4)
    # 找到最大值的索引作为标签
    label = np.argmax(x)  # 返回0,1,2,3
    return x, label

# 随机生成一批样本
# def build_dataset(total_sample_num):
#     X = []
#     Y = []
#     for i in range(total_sample_num):
#         x, y = build_sample()
#         X.append(x)
#         Y.append(y)
#     # print(X)
#     # print(Y)
#     return torch.FloatTensor(X), torch.FloatTensor(Y)
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 注意：这里不加[]，直接存标量
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签用LongTensor

# 测试代码·
# 用来测试每轮模型的准确率
#evaluate(model)：定义一个评估函数，接收训练好的模型
##这里需要修改  一会想想怎么改
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # correct, wrong = 0, 0
    # with torch.no_grad():
    #     y_pred = model(x)  # 模型预测 model.forward(x)
    #     for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
    #         if float(y_p) < 0.5 and int(y_t) == 0:
    #             correct += 1  # 负样本判断正确
    #         elif float(y_p) >= 0.5 and int(y_t) == 1:
    #             correct += 1  # 正样本判断正确
    #         else:
    #             wrong += 1
    # print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    # return correct / (correct + wrong)
       # 统计各类别样本数
    print("各类别样本分布:")
    for i in range(4):
        count = (y == i).sum().item()
        print(f"  类别{i}: {count}个")
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，shape: (100, 4) 概率值
        pred_labels = torch.argmax(y_pred, dim=1)  # 预测类别
        correct = (pred_labels == y).sum().item()
    
    accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 4  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())
    print("模型加载成功")
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()  # 获取预测类别
        prob = res[pred_class].item()  # 获取该类的概率
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class, prob))  # 打印结果


if __name__ == "__main__":
    main()
