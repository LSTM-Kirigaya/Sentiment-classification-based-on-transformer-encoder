import torch
import torch.nn as nn
from random import shuffle
import numpy as np
import pandas as pd
import math
from parl.utils import logger
from sklearn.metrics import accuracy_score

from process import DataLoader
from neural_network import SentimentFourClassifier

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
device = "cpu"

# Leslie的Triangle2学习率衰减方法
def Triangular2(T_max, gamma):
    def new_lr(step):
        region = step // T_max + 1    # 所处分段
        increase_rate = 1 / T_max * math.pow(gamma, region - 2)    # 增长率的绝对值
        return increase_rate * (step - (region - 1) * T_max) if step <= (region - 0.5) * T_max else - increase_rate * (step - region * T_max)
    return new_lr

# 创建数据加载器
# 获取数据
data_dict = torch.load("./data/process_index_seq.rar")

# 获取数据字典后，先打乱，再分割
index_seqs = data_dict["index_seqs"]         # 文本的index序列
labels = data_dict["labels"]                          # 文本对应的标签
voc_dict = data_dict["voc_dict"]
data_num = len(index_seqs)                         # 数据数量
split_ratio = 0.8                                             # 分割比例
split_index = int(data_num * split_ratio)     # 分割的边界的index

# 打乱数据
shuffle_index = np.arange(len(index_seqs))
shuffle(shuffle_index)
X = np.array(index_seqs)
y = np.array(labels)

# 分割数据，得到训练集和测试集列表
train_X = X[shuffle_index[:split_index]].tolist()
train_y = y[shuffle_index[:split_index]].tolist()
test_X = X[shuffle_index[split_index:]].tolist()
test_y = y[shuffle_index[split_index:]].tolist()

# start_time = time()
# train_loader = DataLoader(data=train_X,
#                           target=train_y,
#                           batch_size=64,
#                           shuffle=True)
#
# for batch in train_loader:
#     b_x, b_x_length, b_y = batch
#     print(b_x[0])
#     print(b_x_length[0])
#     print(b_y[0])
#     print([voc_dict["index2word"].get(index, "UNK") for index in b_x[0].tolist()])
#     break
#
# print("take time:", time() - start_time)

# 定义超参
LEARNING_RATE = 1e-4
BATCH_SIZE = 64

print_interval = 10
save_interval = 50

# 获取网络
classifier = SentimentFourClassifier(vocab_dim=voc_dict["num_words"])
# 获取优化器
optimizer = torch.optim.RMSprop(classifier.parameters(), lr=LEARNING_RATE)
# 获取损失函数
loss_func = nn.CrossEntropyLoss(reduction="mean")

global_step = 0
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

for epoch in range(10):
    print("-" * 20)
    print("Epoch : ", epoch)
    # 获取加载器
    loader = DataLoader(data=train_X,
                        target=train_y,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

    classifier.train()
    for batch in loader:
        global_step += 1
        b_x_t, b_x_length, b_y_t = batch

        output = classifier(b_x_t)
        predict_label = torch.argmax(output, 1)
        loss = loss_func(output, b_y_t.flatten())
        acc = accuracy_score(predict_label.flatten(), b_y_t.flatten())
        # 剃度清空
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 将信息记录在列表中
        train_loss_all.append(loss.item())
        train_acc_all.append(acc)

        if global_step % print_interval == 0:
            logger.info("Epoch : {}\tLoss : {:.4f}\tAcc : {:.4f}\tStep : {}".format(
                epoch, train_loss_all[-1], train_acc_all[-1], global_step
            ))

        if global_step % save_interval == 0:
            torch.save({
                "model_dict" : classifier.state_dict(),
                "optimizer" : optimizer,
                "voc_dict" : voc_dict
            }, f"./data/{global_step}_model.rar")

    # 测试
    classifier.eval()
    loader = DataLoader(data=test_X,
                        target=test_y,
                        batch_size=256,
                        shuffle=True)
    for batch in loader:
        b_x_t, b_x_length, b_y_t = batch
        break

    output = classifier(b_x_t)
    predict_label = torch.argmax(output, 1)
    loss = loss_func(output, b_y_t.flatten())
    acc = accuracy_score(predict_label.flatten(), b_y_t.flatten())
    test_loss_all.append(loss.item())
    test_acc_all.append(acc)
    logger.info("Epoch : {}\tLoss : {:.4f}\tAcc : {:.4f}\tStep : {}".format(
        epoch, test_loss_all[-1], test_acc_all[-1], global_step
    ))

train_pd = pd.DataFrame({
    "train_loss" : train_loss_all,
    "train_acc" : train_acc_all
})

test_pd = pd.DataFrame({
    "test_loss" : test_loss_all,
    "test_acc" : test_acc_all
})

train_pd.to_csv("./data/train_log.csv", index=None)
test_pd.to_csv("./data/test_log.csv", index=None)