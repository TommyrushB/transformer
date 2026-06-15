import os, sys
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformer.train import Batch, LabelSmoothing, rate, run_epoch, TrainState
import numpy as np
import pandas as pd
import torch.nn as nn
from transformer.helpers import clones, LayerNorm
import copy
from transformer.network import (MultiHeadedAttention, PositionwiseFeedForward,
                                  PositionalEncoding,EncoderDecoder, Encoder,
                                 EncoderLayer, Decoder, DecoderLayer, Embeddings,
                                 Generator,attention)
from transformer.train import Batch, LabelSmoothing, rate, run_epoch, TrainState, SimpleLossCompute, greedy_decode
from torch.utils.data import TensorDataset
from prepare import generate_new_resultdir
import math

#加载训练集
load_data = []
for i in range(1,201):
    if i == 22:
        pass
    else:
        data = pd.read_csv("data/%d.csv"%i)
        load_data.append(data)
load_data = np.stack(load_data)
y = pd.read_csv("trainLabels.csv",index_col=0).to_numpy()
y = np.concatenate([y[:21], y[22:]])
load_data = torch.Tensor(load_data)  #torch.Size([199, 55, 442])
y = torch.Tensor(y)  #torch.Size([199, 198])
load_data = torch.diff(load_data,dim=1)#torch.Size([199, 54, 442])  做差分
data_tgt = load_data[:,:,:198].transpose(1,2)#torch.Size([199, 198, 54])
data_src = load_data[:,:,198:].transpose(1,2)#torch.Size([199, 244, 54])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#模型构建    将244列特征作为编码器输入，将198个公司数据作为解码器输入，没有掩码
class Encoder_fix1(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder,linear):
        super(Encoder_fix1, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = linear

    def forward(self, src, tgt, src_mask, tgt_mask):

        return self.linear(self.decoder(tgt,self.encoder(src, src_mask), src_mask,  tgt_mask))
class Encoder_fix2(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, linear):
        super(Encoder_fix2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.linear = linear

    def forward(self, src, tgt, src_mask, tgt_mask):

        return  self.linear(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_kaggle_model(
    m=1,#  1：没有进行位置编码，2有位置编码
    N=2,
    d_model=54,
    d_ff=54,
    h=9,
    dropout=0.1,):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    if m == 1:
        model = Encoder_fix1(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(
                          nn.Linear(54, 1),
                        )
    )
    elif m == 2:
        model = Encoder_fix2(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            c(position),
            c(position),
            nn.Sequential(
                          nn.Linear(54, 1),
                        )
        )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
def loss_fn(output, target):
    return nn.MSELoss()(output, target)


#建立dataloader
dataset = TensorDataset(data_src,data_tgt, y)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False, )

RESULT_DIR = generate_new_resultdir(framework="kaggle")
d_model = 54
epoch = 10000
mask =None
log = []

model = make_kaggle_model().to(device)
file = f'company.pt'
MODEL_PT_FILEPATH = os.path.join(RESULT_DIR, file)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
)
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(
        step, d_model, factor=0.1, warmup=500
    ), )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',           # 最小化损失
#     factor=0.9,           # 每次降低50%
#     patience=5,           # 5个epoch不改善就降低
#     threshold=1e-4,       # 最小改善阈值
#     cooldown=2,           # 降低后等待2个epoch
#     min_lr=1e-7,          # 最小学习率
#     verbose=True          # 打印信息
# )
model.train()
for i in range(1,epoch+1):

    for src,tgt,y in train_loader:

        src, tgt,y= src.to(device), tgt.to(device),y.to(device)

        optimizer.zero_grad()
        out = model(src,tgt,mask,mask)
        out.squeeze_()
        # print(out.shape,tgt[:,:,53].shape)
        # sys.exit(1)
        bias = out+tgt[:,:,53]


        loss = loss_fn(bias,y)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    #scheduler.step(loss)
    print(f'epoch:{i},loss:{loss:.4f}')
    if loss < 0.01:
        break
torch.save(model.state_dict(), MODEL_PT_FILEPATH)
del model

#加载验证集
validate_data = []
for i in range(201,511):
    if i == 22:
        pass
    else:
        data = pd.read_csv("data/%d.csv"%i)
        validate_data.append(data)
validate_data = np.stack(validate_data)
validate_data = torch.Tensor(validate_data).to(device)
validate_data = torch.diff(validate_data,dim=1)
validate_data_tgt = validate_data[:,:,:198].transpose(1,2)
validate_data_src = validate_data[:,:,198:].transpose(1,2)

#计算结果
result = []
model = make_kaggle_model().to(device)

copy_model = copy.deepcopy(model)
file = f'company.pt'
result_dir = os.path.join(RESULT_DIR,file)

model.load_state_dict(torch.load(result_dir,map_location=device,weights_only=True))

model.eval()
with torch.no_grad():
    out = model(validate_data_src,validate_data_tgt,mask,mask)
    out.squeeze_()
    bias = out+validate_data_tgt[:,:,53]
result.append(bias)

del copy_model
torch.cuda.empty_cache()

#结果保存为文件
a = torch.vstack([t.cpu() for t in result])
b = pd.DataFrame(a,dtype=np.float64)
file1 = pd.read_csv('sampleSubmission.csv',engine='python')
file1 = file1.astype({col: 'float64' for col in file1.columns if col != 'FileId'})
file1.iloc[:, 1:] = b
file1.to_csv('result/sampleSubmission.csv', index=False)