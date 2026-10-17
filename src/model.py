import math
import torch
import torch.nn as nn
from config import *

#旋转位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D, max_len=MAX_LENGTHS):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model//2):
                angle = pos / (10000.0 ** (2 * i / d_model))
                pe[pos, 2 * i] = math.sin(angle)
                pe[pos, 2 * i + 1] = math.cos(angle)
        # 固定位置表，不参与梯度更新
        self.register_buffer('pe', pe)


    def forward(self, x):
        rows = []
        for pos in range(len(x)):
            rows.append(x[pos] + self.pe[pos]) #两个[D]向量逐元素相加
        return torch.stack(rows) #把多行[D]叠成[句长，D]

#Embedding
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model=D,max_len=MAX_LENGTHS):
        super(TokenEmbedding, self).__init__()
        #创建形状为[vocab_size,D]的词向量表，每个Token ID 对应一行
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model,max_len)

    def forward(self,tokens_id,d_model = D):
        # 输入为一维 Token ID 张量，形状为[L]
        # 按ID查出每个Token的词向量，得到[L,D]，再乘以sqrt(D)，增强语义信息。
        x= self.embedding(tokens_id) * math.sqrt(d_model)
        return self.positional_encoding(x)

#注意力计算公式
def attention(q,k,v,causal = False):
    #参见论文中注意力分数计算公式，[Lq,Dh] @ [Dh,Lk] -> [Lq,Lk]，每一行都是一个 Query 分数。
    scores = q @ k.T / math.sqrt(q.shape[-1])

    if causal:
        #因果掩码为真，使用下三角矩阵掩码
        mask = torch.triu(torch.ones_like(scores,dtype = torch.bool),diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    #沿着 Key 方向做softmax；被屏蔽的位置权重为0

    weights = torch.softmax(scores, dim = -1) #[Lq,Lk]
    return weights @ v #[Lq,Lk] @ [Lk,Dh] -> [Lq,Dh]

#单个注意力头
class AttentionHead(nn.Module):
    def __init__(self,heads = HEADS,d_model=D):
        super(AttentionHead, self).__init__()
        head_dim = d_model // heads #单个注意力头维度Dh
        #分别学习Q,K,V投影，将每个位置从D维映射到Dh维
        self.q = nn.Linear(d_model, head_dim, bias = False)
        self.k = nn.Linear(d_model, head_dim, bias = False)
        self.v = nn.Linear(d_model, head_dim, bias = False)

    def forward(self,query,key,value,causal = False):
        q=self.q(query) #[Lq,D] -> [Lq,Dh]
        k=self.k(key)   #[Lk,D] -> [Lk,Dh]
        v=self.v(value) #[Lk,D] -> [Lk,Dh]
        #用Q,K计算权重，再汇总到V上，输出[Lq,Dh]
        return attention(q,k,v,causal=causal)

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,heads = HEADS,d_model=D,d_head=D):
        super(MultiHeadAttention, self).__init__()
        #每个头分别学习自己的Q,K,V投影，参数相互独立
        self.heads = nn.ModuleList([AttentionHead() for _ in range(heads)])
        #对拼接后的特征做输出投影，融合各个头的信息，维度保持为D
        self.output_proj = nn.Linear(d_model, d_head)

    def forward(self,query,key,value,causal = False):
        output = []
        #所有头读取相同输入，各自得到一个[Lq,Dh]的结果
        for head in self.heads:
            output.append(head(query,key,value,causal=causal))
        #沿特征维度拼接 [Lq,Dh*heads] -> [Lq,D]
        joined = torch.cat(output,dim = -1)
        return self.output_proj(joined)

#末尾FFN线性映射
class FeedForward(nn.Module):
    def __init__(self,d_model=D,d_ff=D_FF,dropout=0.1):
        super(FeedForward, self).__init__()
        self.netowrk = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.netowrk(x)

#单层Encoder
class EncoderLayer(nn.Module):
    def __init__(self,d_model=D):
        super(EncoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention()  #多头注意力
        self.ffn = FeedForward()   #前馈神经网络
        self.norm1 = nn.LayerNorm(d_model) #自注意力残差相加后的层归一化
        self.norm2 = nn.LayerNorm(d_model) #FFN 残差相加后的层归一化

    def forward(self,x):
        #自注意力
        #输入X已经过Embedding,维度为 [S,D]
        x = self.norm1(x + self.self_mha(x,x,x))

        #FNN -》 AddNorm；残差使用进入FFN前的X
        x = self.norm2(x + self.ffn(x))
        return x

#堆叠Encoder
class Encoder(nn.Module):
    def __init__(self,num_layers = NUM_LAYERS):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x  #最终输出[S,D]

#单层Decoder
class DecoderLayer(nn.Module):
    def __init__(self,d_model=D):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention()  #带掩码的多头注意力
        self.cross_mha = MultiHeadAttention() #交叉多头注意力
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,x,encoder_output):
        #因果掩码自注意力
        x = self.norm1(x + self.self_mha(x,x,x,causal=True))
        #交叉注意力,k,v来自编码器，q来自已生成的上文
        x = self.norm2(x + self.cross_mha(x,encoder_output,encoder_output))
        # FFN
        x = self.norm3(x + self.ffn(x))
        return x #最终输出

#堆叠Decoder
class Decoder(nn.Module):
    def __init__(self,num_layers = NUM_LAYERS):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self,x,encoder_output):
        for layer in self.layers:
            x = layer(x,encoder_output)
        return x #最终输出[T,D]

class Transformer(nn.Module):
    def __init__(self,source_words,target_words,num_layers = NUM_LAYERS,d_model = D):
        super(Transformer, self).__init__()
        #两种语言分别使用自己内部的词向量表
        #TokenEmbedding 内部完成查词向量 -> 乘以sqrt(D) ->位置编码
        self.source_embedding = TokenEmbedding(len(source_words))
        self.target_embedding = TokenEmbedding(len(target_words))

        #Encoder 编码英文，Decoder 根据中文前缀读取英文编码结果
        self.encoder = Encoder(num_layers)
        self.decoder = Decoder(num_layers)

        #将每个中文位置的 D 维表示，映射为中文词表中各 Token 分数
        self.output_projection = nn.Linear(d_model,len(target_words))

    def encode(self,source):
        #source 为英文 Token ID,维度大小为S
        return self.encoder(self.source_embedding(source))

    def decode(self,target,encoder_output):
        #target 为中文 Token ID,维度大小为T
        x= self.target_embedding(target) #查词向量，缩放，加位置编码
        x = self.decoder(x,encoder_output)#因果掩码自注意力和交叉注意力，输出维度与输入维度一致
        #输出为logits:[T,中文词表大小]，每一行用于预测对应位置的心意啊一个Token
        return self.output_projection(x)

    def forward(self,source,target):
        encoder_output = self.encode(source)
        return self.decode(target,encoder_output)




