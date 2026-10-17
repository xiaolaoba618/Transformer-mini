import torch
from config import DEVICE
en_cn_pairs = [
    ("there is a cat under the table", "桌子 下面 有 一只 猫"),
    ("there is a dog on the chair", "椅子 上面 有 一只 狗"),
    ("there is a cat on the chair", "椅子 上面 有 一只 猫"),
    ("there is a dog under the table", "桌子 下面 有 一只 狗"),
    ("there is a small cat on the table", "桌子 上面 有 一只 小猫"),
    ("there is a small dog under the chair", "椅子 下面 有 一只 小狗"),
    ("there is a black cat under the chair", "椅子 下面 有 一只 黑色 猫"),
    ("there is a black dog on the table", "桌子 上面 有 一只 黑色 狗"),
]

# 1. 收集全部英文源单词、中文目标单词
source_words_set = set()
target_words_set = set()

for en_sent, cn_sent in en_cn_pairs:
    # 英文按空格切分
    en_tokens = en_sent.split()
    for w in en_tokens:
        source_words_set.add(w)

    # 中文按空格切分
    cn_tokens = cn_sent.split()
    for w in cn_tokens:
        target_words_set.add(w)

# 去重后转列表
source_words = sorted(list(source_words_set))
# target_words 前面插入 <BOS> <EOS>
target_words = ["<BOS>", "<EOS>"] + sorted(list(target_words_set))

# 列表用于ID -> Token ，字典用于Token -> ID
source_ids = {word:i for i , word in enumerate(source_words)}
target_ids = {word:i for i , word in enumerate(target_words)}

BOS = target_ids['<BOS>']
EOS = target_ids['<EOS>']


#Teacher Forcing 预处理
decoder_input = target_words[:-1] #输入去掉末尾EOS
labels = target_words[1:]  #输出标签去掉头部BOS




# 将句子转为一维整数张量，中文目标两端加入BOS,EOS
examples = []

for en_sent, cn_sent in en_cn_pairs:
    source = [source_ids[w] for w in en_sent.split()]
    target = [BOS] + [target_ids[w] for w in cn_sent.split()] + [EOS]
    examples.append((
        torch.tensor(source,dtype = torch.long).to(DEVICE),
        torch.tensor(target,dtype = torch.long).to(DEVICE),
    ))




