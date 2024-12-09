import torch
import torch.nn as nn
import numpy as np


class BLSTMConversionModel(nn.Module):
    """
    Conversion model based on BLSTM
    """

    def __init__(self, in_channels, out_channels, lstm_hidden):
        """
        :param in_channels: input feature dimension,
                            usually (bnfs_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        """
        super(BLSTMConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.blstm1 = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden, bidirectional=True
        )
        self.blstm2 = nn.LSTM(
            input_size=lstm_hidden * 2, hidden_size=lstm_hidden, bidirectional=True
        )
        self.out_projection = nn.Linear(
            in_features=2 * lstm_hidden, out_features=out_channels
        )

    def forward(self, x):
        """
        :param x: [time, batch, features]
        :return: [time, batch, features]
        """
        # pass to the 1st BLSTM layer
        blstm1_out, _ = self.blstm1(x)
        # pass to the 2nd BLSTM layer
        blstm2_out, _ = self.blstm2(blstm1_out)
        # project to the output dimension
        outputs = self.out_projection(blstm2_out)
        return outputs


class BLSTMResConversionModel(nn.Module):
    """
    Conversion model based on BLSTM with ResidualNet, you need to
    define your ResidualNet Module to be used in this Module.
    """

    def __init__(self, in_channels, out_channels, lstm_hidden, other_params=None):
        """
        :param in_channels: input feature dimension,
                            usually (bnf_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        :param other_params: other parameters you need to define resnet
        """
        super(BLSTMResConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.blstm1 = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden, bidirectional=True
        )
        self.blstm2 = nn.LSTM(
            input_size=lstm_hidden * 2, hidden_size=lstm_hidden, bidirectional=True
        )
        self.out_projection = nn.Linear(
            in_features=2 * lstm_hidden, out_features=out_channels
        )
        self.resnet = ResidualNet(out_channels, other_params)

    def forward(self, x):
        """
        :param x: [time, batch, features]
        :return: 最终输出 [time, batch, features]
        """
        # 通过第一个 BLSTM 层
        blstm1_out, _ = self.blstm1(x)
        # 通过第二个 BLSTM 层
        blstm2_out, _= self.blstm2(blstm1_out)
        # 投影到输出维度
        initial_outs = self.out_projection(blstm2_out)
        # 计算残差
        residual = self.resnet(initial_outs)
        # 将残差与初始输出相加，得到最终输出
        final_outs = initial_outs + residual
        return final_outs


class ResidualNet(nn.Module):
    """
    用于生成预测梅尔频谱残差的残差网络。
    """
    def __init__(self, channels, other_params):
        super(ResidualNet, self).__init__()
        self.channels = channels
        # 定义两层全连接层和激活函数
        self.fc1 = nn.Linear(in_features=channels, out_features=channels)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=channels, out_features=channels)
        
    def forward(self, x):
        """
        计算残差。
        :param x: 输入的初始梅尔频谱 [time, batch, channels]
        :return: 残差 [time, batch, channels]
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out  # 返回残差

class BLSTMToManyConversionModel(nn.Module):
    """
    BLSTM based any-to-many VC model with SPKEmbedding Module.
    You need to define your SPKEmbedding Module, and
    might need to change necessary components in this Module.
    """

    def __init__(self, in_channels, out_channels, num_spk, embd_dim, lstm_hidden):
        """
        :param in_channels: input feature dimension,
                            usually (bnf_dim + f0s_dim) when use bnfs and f0s as inputs
        :param out_channels: mel dimension or your acoustic feature dimnesion
        :param lstm_hidden: parameter dimension
        """
        super(BLSTMToManyConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_spk = num_spk
        self.embd_dim = embd_dim
        self.lstm_hidden = lstm_hidden
        self.spk_embed_net = SPKEmbedding(num_spk=num_spk, embd_dim=embd_dim)
        self.emb_proj1 = nn.Linear(in_features=embd_dim, out_features=in_channels)
        self.emb_proj2 = nn.Linear(in_features=embd_dim, out_features=lstm_hidden * 2)
        self.blstm1 = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden, bidirectional=True
        )
        self.blstm2 = nn.LSTM(
            input_size=lstm_hidden * 2, hidden_size=lstm_hidden, bidirectional=True
        )
        self.out_projection = nn.Linear(
            in_features=2 * lstm_hidden, out_features=out_channels
        )

    def forward(self, x, spk_inds):
        """
        :param x: 输入特征，形状为 [time_steps, batch_size, in_channels]
        :param spk_inds: 目标说话人索引，形状为 [batch_size]
        :return: 预测的梅尔频谱，形状为 [time_steps, batch_size, out_channels]
        """
        # 查找说话人嵌入
        spk_embds = self.spk_embed_net(spk_inds)  # [batch_size, embd_dim]
        # 将说话人嵌入扩展到时间步长度
        spk_embds_time = spk_embds.unsqueeze(0).expand(x.shape[0], -1, -1)  # [time_steps, batch_size, embd_dim]
        # 对嵌入进行线性变换
        spk_embd_proj1 = self.emb_proj1(spk_embds_time)  # [time_steps, batch_size, in_channels]
        spk_embd_proj2 = self.emb_proj2(spk_embds_time)  # [time_steps, batch_size, lstm_hidden * 2]
        # 将嵌入添加到输入特征
        blstm1_inputs = x + spk_embd_proj1  # [time_steps, batch_size, in_channels]
        # 通过第一个 BLSTM 层
        blstm1_outs, _ = self.blstm1(blstm1_inputs)
        # 将嵌入添加到第一个 BLSTM 层的输出
        blstm2_inputs = blstm1_outs + spk_embd_proj2  # [time_steps, batch_size, lstm_hidden * 2]
        # 通过第二个 BLSTM 层
        blstm2_outs, _ = self.blstm2(blstm2_inputs)
        # 投影到输出维度
        outputs = self.out_projection(blstm2_outs)
        return outputs


class SPKEmbedding(nn.Module):
    """
    说话人嵌入模块。
    """
    def __init__(self, num_spk, embd_dim):
        """
        初始化说话人嵌入模块。
        :param num_spk: 目标说话人的数量
        :param embd_dim: 说话人嵌入向量的维度
        """
        super(SPKEmbedding, self).__init__()
        # 定义可学习的嵌入表
        self.embedding_table = nn.Embedding(num_spk, embd_dim)
        
    def forward(self, spk_inds):
        """
        根据说话人索引返回对应的嵌入向量。
        :param spk_inds: 说话人索引，torch.LongTensor 类型，形状为 [batch_size]
        :return: 说话人嵌入，形状为 [batch_size, embd_dim]
        """
        return self.embedding_table(spk_inds)


class CustomToOneConversionModel(nn.Module):
    """
    define your custom any-to-one conversion model here
    """

    def __init__(self, in_channels, out_channels, other_params):
        """
        :param in_channels: input feature dimension (e.g. bnf's dim + F0 dim)
        :param out_channels: output feature dimension (e.g. mel-spectrogram's dim)
        :param other_params: other parameters for your custom defined components
        """
        super(CustomToOneConversionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # define your own conversion model's components below
        # e.g. self.dense_layer = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        """
        :param x: VC input features
        :return: output acoustic features
        """
        # define your model's transform process below
        pass


class Mel2Linear(nn.Module):
    """
    Here defines a module to transform mel-spectrogram into linear-spectrogram
    in a neural way.
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: input mel-spectrogram feature dimensions
        :param out_channels: output linear-spectrogram feature dimensions
        """
        super(Mel2Linear, self).__init__()
        # define your module components below
        # e.g. self.dense_layer = nn.Linear(in_features=in_channels, out_features=out_channels)

    def call(self, x):
        """
        :param x: input mel-spectrogram
        :return: output linear-spectrogram
        """
        # define your transform process below


if __name__ == "__main__":
    # these are just some tests
    x = torch.randn(4, 16, 8)
    mdl = BLSTMConversionModel(in_channels=8, out_channels=9, lstm_hidden=16)
    print(mdl)
    outs = mdl(x)
    print(outs.shape)
