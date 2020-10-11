import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import jieba
import wordcloud
from itertools import zip_longest
from time import time
import random

# 输出图显示为中文
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="D:/编程/font/HGY1_CNKI.TTF")

class vocab(object):
    def __init__(self, name, pad_token, unk_token, stop_words=[]):
        """
        :param name: 词表的名字
        :param pad_token: 填充字符
        :param unk_token: 未知字符
        :param stop_words: 停用词
        """
        self.name = name
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.stop_words = stop_words

        self.is_trim = False     # 数据集是否经过剪枝

        self.word2index = {"PAD" : self.pad_token, "UNK" : self.unk_token}
        self.word2count = {"PAD" : 10, "UNK" : 10}
        self.index2word = {self.pad_token : "PAD", self.unk_token : "UNK"}
        self.num_words = 2      # 目前字典中已经存在的单词数量

    # 增加一个分词的逻辑
    def addWord(self, word):
        # 如果需要添加的词语出现在了停用词列表，直接去除
        if word in self.stop_words:
            return
        # 如果word是数字，则去除
        if word.isnumeric():
            return
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1

    # 添加一个句子的逻辑
    def addSentence(self, sentence):
        for word in jieba.lcut(sentence):
            self.addWord(word)

    # 剪枝的逻辑只保留min_count以上的分词
    def trim(self, min_count):
        # 只能剪枝一次，若本词表已经剪枝过，直接返回
        if self.is_trim:
            raise ValueError(f"vocab \"{self.name}\" has been trimmed, ensure method trim only be used once!")
            return

        self.is_trim = True

        keep_words = []
        keep_num = 0            # 保留的分词的总数
        drop_words = []
        drop_num = 0            # 丢弃的分词的总数

        for word, count in self.word2count.items():
            if count < min_count:
                drop_num += 1
                # 由于后面是通过对keep_word列表中的数据逐一统计，所以需要对count>1的单词重复填入
                drop_words.append(word)
            else:
                keep_num += 1
                for _ in range(count):
                    keep_words.append(word)

        print("keep word: {} / {}={:.4f}".format(
            keep_num, (keep_num + drop_num), keep_num / (keep_num + drop_num)
        ))
        # 去重后使用词云可视化丢弃的单词
        w = wordcloud.WordCloud(width=1500, height=1200, font_path="msyh.ttc")
        w.generate(" ".join(drop_words))
        w.to_file("./data/process/dropped_words.png")

        # 重构字典
        self.word2index = {"PAD": self.pad_token, "UNK": self.unk_token}
        self.word2count = {"PAD" : 10, "UNK" : 10}
        self.index2word = {self.pad_token: "PAD", self.unk_token: "UNK"}
        self.num_words = 2  # 目前字典中已经存在的单词数量
        for word in keep_words:
            self.addWord(word)

    # 载入数据
    def load_data(self, path):
        table = pd.read_csv(path)
        for line_index in range(table.shape[0]):
            self.addSentence(table.review[line_index])

    # 将一句话转化成index序列，并返回
    # 首先去除停用词，再使用pad_token代替不在词表中的词
    def sentence2IndexSeq(self, sentence):
        index_seq = []
        for word in jieba.lcut(sentence):
            if word in self.stop_words:
                continue
            if word.isnumeric():
                continue
            else:
                index_seq.append(self.word2index.get(word, self.unk_token))
        return index_seq

    # 将原数据转换成index序列，并存入target_path中
    def transformOriginDataset(self, path, target_path, voc):
        table = pd.read_csv(path)
        index_seqs = []
        labels = []
        for index in range(table.shape[0]):
            # 将单句话转换成index序列
            index_seq = self.sentence2IndexSeq(table.review[index])
            index_seqs.append(index_seq)
            labels.append(table.label[index])

        torch.save({
            "index_seqs" : index_seqs,
            "labels" : labels,
            "voc_dict": voc.__dict__
        }, target_path)

# 将一个batch的index序列转换成填充padded的规整数据
def batchIndexSeq2paddedTensor(batch):
    """
    :param batch: 二维列表，不等长
    :return: 填充pad的tensor和描述每个数据长度的tensor
    """
    length_tensor = torch.tensor([len(index_seq) for index_seq in batch])
    zipped_list = list(zip_longest(*batch, fillvalue=0))
    padded_tensor = torch.tensor(zipped_list).t()
    return padded_tensor, length_tensor

# 数据加载器
def DataLoader(data, target, batch_size, shuffle=True):
    indexes = np.arange(len(data))
    if shuffle:
        random.shuffle(indexes)
        b_x = []   # 一个batch的data
        b_y = []  # 一个batch的target
    for i in indexes:
        b_x.append(data[i])
        b_y.append(target[i])
        # 达到数量要求，就把数据yield出去
        if len(b_x) % batch_size == 0:
            # b_x要pad填成矩阵
            b_x_t, b_x_length = batchIndexSeq2paddedTensor(b_x)
            b_y_t = torch.LongTensor(b_y)

            b_x, b_y = [], []
            yield [
                b_x_t, b_x_length, b_y_t
            ]

if __name__ == "__main__":
    start_time = time()
    PAD_token = 0
    UNK_token = 1

    data_path = "D:/编程/数据集/simplifyweibo_4_moods/simplifyweibo_4_moods.csv"
    # 读入停用词表
    stop_words = open("./data/baidu_stop_words.txt", "r", encoding="utf-8").read().split("\n")

    # 获取词典
    voc = vocab(name="simplifyweibo_4_moods",
                pad_token=PAD_token,
                unk_token=UNK_token,
                stop_words=stop_words)

    print("load data to construct vocab...")
    voc.load_data(path=data_path)

    print("trimming data...")
    voc.trim(min_count=3)

    # 获取count前30的分词
    most_common = sorted(voc.word2count, key=lambda x : voc.word2count[x], reverse=True)[:30]
    word_count = pd.DataFrame({
        "word" : most_common,
        "count" : [voc.word2count[word] for word in most_common]
    })
    print(most_common)

    # word_count.plot(x="word", y="count", kind="bar", legend=True, figsize=[12, 8])
    # plt.xticks(rotation=90, fontproperties=fonts, size=10)
    # plt.grid(True)
    # plt.show()

    # 将数据转换成index序列，并存入文件中
    print("save processed data to \"./data/process_index_seq.rar\"")
    voc.transformOriginDataset(path=data_path, target_path="./data/process_index_seq.rar", voc=voc)

    total_time = time() - start_time
    print("processing completed. It takes {:.0f}mins {:.2f}seconds".format(total_time / 60, total_time % 60))
