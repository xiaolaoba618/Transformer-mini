import random
import torch

#固定随机种子
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

D = 32 #每个token的向量维度
HEADS = 2 #注意力头数;D必须能被HEADS整除
D_FF =64      #FFN中间层的维数
NUM_LAYERS = 2 #整个Transformer 堆叠数量，即Encoder 和 Decoder 各自堆叠的层数
MAX_LENGTHS = 32 #位置编码表支持的最大 Token 数，即单个句子最大长度
TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 5
MIN_LR = 1e-5
MAX_LR = 0.001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #定义训练设备