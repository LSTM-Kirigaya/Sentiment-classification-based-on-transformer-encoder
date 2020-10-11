import torch
import torch.nn as nn
import numpy as np

# 位置嵌入
class PositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout = 0.1,max_len = 500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

# 情感分类网络
class SentimentFourClassifier(nn.Module):
    def __init__(self, vocab_dim, embedding_dim=512, output_dim=4, transformer_header_num=4,
                 dim_feedforward=2048, transformer_encoder_norm=None,  transformer_layer_num=5,
                 gru_hidden_dim=1024, gru_layers=2, gru_dropout=0.1):
        """
        :param vocab_dim: 词表的长度
        :param embedding_dim: 词向量嵌入维度
        :param output_dim: 输出维度，也就是分类种数
        :param transformer_header_num: transformer encoder的注意力头数量
        :param dim_feedforward: transformer encoder中的前馈网络的层数
        :param transformer_encoder_norm: transformer encoder最终的norm
        :param transformer_layer_num: transformer encoder layer的数量
        :param gru_hidden_dim: gru的隐层维度
        :param gru_layers: gru层数
        :param gru_dropout:gru层的丢弃概率
        """
        super(SentimentFourClassifier, self).__init__()

        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.transformer_header_num = transformer_header_num
        self.dim_feedforward = dim_feedforward
        self.transformer_encoder_norm = transformer_encoder_norm
        self.transformer_layer_num = transformer_layer_num
        self.gru_layers = gru_layers

        # 词向量嵌入算子
        self.embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim)
        # 位置嵌入算子
        self.position_encoding = PositionEncoding(d_model=embedding_dim)
        # 任意一个transformer_encoder_layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                   nhead=transformer_header_num,
                                                   dim_feedforward=dim_feedforward,
                                                   activation="relu")
        # 获得多层transformer编码器映射
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                    num_layers=transformer_layer_num,
                                                    norm=transformer_encoder_norm)

        # gru
        self.gru1 = nn.GRU(input_size=embedding_dim,
                          hidden_size=embedding_dim,
                          dropout=(0 if gru_layers == 1 else gru_dropout),
                          num_layers=gru_layers,
                          bidirectional=True,
                          batch_first=True)

        self.gru2 = nn.GRU(input_size=embedding_dim,
                          hidden_size=embedding_dim,
                          dropout=(0 if gru_layers == 1 else gru_dropout),
                          num_layers=gru_layers,
                          bidirectional=True,
                          batch_first=True)

        self.gru3 = nn.GRU(input_size=embedding_dim,
                          hidden_size=gru_hidden_dim,
                          dropout=(0 if gru_layers == 1 else gru_dropout),
                          num_layers=gru_layers,
                          bidirectional=True,
                          batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, 512),
            nn.Linear(512, 128)
        )

        self.classify = nn.Linear(128, 4)

    def forward(self, input_seq, input_length):
        # x : index序列  [B, max_length]
        out = self.embedding(input_seq)     # [B, max_length, embedding_dim]
        out = self.position_encoding(out)      # [B, max_length, embedding_dim]
        out = self.transformer_encoder(out)    # [B, max_length, embedding_dim]

        # output1, hidden1 = self.gru1(out, None)
        # # output : [B, max_length, embedding * 2]
        # # hidden : [gru_layers * 2, B, hidden_dim]
        # # 残差连接1
        # residual_out1 = out + output1[ : , : , : self.embedding_dim] + output1[ : , : ,self.embedding_dim : ]     # [B, max_length, embedding_dim]
        # # 残差连接2
        # output2, hidden2 = self.gru2(residual_out1, None)
        # residual_out2 = residual_out1 + output2[ : , : , : self.embedding_dim] + output2[ : , : ,self.embedding_dim : ]         # [B, max_length, embedding_dim]
        # # 通过最后一个gru将混合特征映射映射到一个高维向量
        # _, hidden3 = self.gru3(residual_out2, None)      # hidden : [gru_layers * 2, B, gru_hidden_dim]
        # feature_map = torch.sum(hidden3, dim=0)         #  [B, gru_hidden_dim]

        # 压缩序列
        packed = nn.utils.rnn.pack_padded_sequence(input=out, lengths=input_length, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru3(packed, None)
        feature_map = torch.sum(hidden, dim=0)

        # 将混合特征向量放入全连接层+softmax分类
        out = self.fc(feature_map)
        output = self.classify(out).softmax(dim=1)

        return output

if __name__ == "__main__":
    BATCH_SIZE = 64
    MAX_LENGTH = 64
    EMBEDDING_DIM = 128

    x = torch.ones(128, 80)
    length = torch.ones(128)
    print(x.shape)

    net = SentimentFourClassifier(vocab_dim=5000,
                                  embedding_dim=EMBEDDING_DIM)

    output = net(x.type(torch.int64), length.type(torch.int64))
    print(output.shape)