import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# 1. 设定超迷你参数
B = 1         # 1 条序列
T = 4         # 上下文长度 4 个字，例如输入 "step"
n_embd = 8    # 每个字符用 8 维向量表示
n_head = 2    # 2 个注意力头
head_size = n_embd // n_head # 每个头分到 4 维 (8 // 2 = 4)
vocab_size = 65

print("="*60)
print(f"【超微参数】B={B}, T={T}, n_embd={n_embd}, n_head={n_head}, head_size={head_size}")
print("="*60)

# 模拟输入：4 个字符的 token id
idx = torch.tensor([[10, 25, 4, 18]]) # (B=1, T=4)
print(f"\n1. 输入 token 索引 idx: shape = {idx.shape}")
print(f"   内容: {idx.tolist()}")

# 2. Embedding 层
token_embedding_table = nn.Embedding(vocab_size, n_embd)
position_embedding_table = nn.Embedding(T, n_embd)

tok_emb = token_embedding_table(idx) # (B, T, C) -> (1, 4, 8)
pos_emb = position_embedding_table(torch.arange(T)) # (T, C) -> (4, 8)
x = tok_emb + pos_emb # (1, 4, 8) (pos_emb 自动广播给 batch)

print(f"\n2. 嵌入层 (Embedding):")
print(f"   tok_emb shape: {tok_emb.shape}  (查表把每个字变成 8 维)")
print(f"   pos_emb shape: {pos_emb.shape}  (给 0,1,2,3 位置各发一个 8 维位置编码)")
print(f"   x (相加后) shape: {x.shape}  (每个字既有身份信息，又有位置信息)")

# 3. 单个注意力头（Head 0）内部单步拆解
key = nn.Linear(n_embd, head_size, bias=False)   # 8 -> 4
query = nn.Linear(n_embd, head_size, bias=False) # 8 -> 4
value = nn.Linear(n_embd, head_size, bias=False) # 8 -> 4
tril = torch.tril(torch.ones(T, T))

k = key(x)   # (1, 4, 4)
q = query(x) # (1, 4, 4)
v = value(x) # (1, 4, 4)

print(f"\n3. 单头注意力 (Head 0) 内部视角:")
print(f"   k (Key)   shape: {k.shape} (8 维特征压缩为 4 维键)")
print(f"   q (Query) shape: {q.shape} (8 维特征压缩为 4 维查询)")
print(f"   v (Value) shape: {v.shape} (8 维特征压缩为 4 维干货内容)")

# 计算注意力打分
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (1, 4, 4) @ (1, 4, 4) -> (1, 4, 4)
print(f"   q @ k.T   shape: {wei.shape} (得到 4x4 的两两相关性打分表)")

wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
print(f"   wei (softmax后) shape: {wei.shape} (下三角权重，未来位置全为 0)")

out_h0 = wei @ v # (1, 4, 4) @ (1, 4, 4) -> (1, 4, 4)
print(f"   out (wei @ v) shape: {out_h0.shape} (根据打分，把前面字的 v 加权混在一起)")

# 4. 多头注意力拼接 (Multi-Head)
# 假设 Head 1 也算出了同样形状的 (1, 4, 4)
out_h1 = torch.randn(1, 4, 4)
out_multi = torch.cat([out_h0, out_h1], dim=-1) # (1, 4, 8)
proj = nn.Linear(n_embd, n_embd)
sa_out = proj(out_multi) # (1, 4, 8)

print(f"\n4. 多头拼接 (MultiHeadAttention):")
print(f"   cat([Head0, Head1], dim=-1) shape: {out_multi.shape} (拼回 8 维)")
print(f"   proj 线性投影输出 shape: {sa_out.shape}")

# 5. 残差连接 1
x = x + sa_out
print(f"\n5. 第 1 次残差相加: x = x + sa_out -> shape: {x.shape}")

# 6. 前馈网络 (FeedForward)
ffwd_linear1 = nn.Linear(n_embd, 4 * n_embd) # 8 -> 32
relu = nn.ReLU()
ffwd_linear2 = nn.Linear(4 * n_embd, n_embd) # 32 -> 8

ff_mid = relu(ffwd_linear1(x)) # (1, 4, 32)
ff_out = ffwd_linear2(ff_mid)  # (1, 4, 8)

print(f"\n6. 前馈网络 (FeedForward / 独立思考):")
print(f"   扩宽 4 倍: Linear(8 -> 32) -> shape: {ff_mid.shape}")
print(f"   压回原宽: Linear(32 -> 8) -> shape: {ff_out.shape}")

# 7. 残差连接 2
x = x + ff_out
print(f"\n7. 第 2 次残差相加: x = x + ff_out -> shape: {x.shape}")

# 8. 最终输出头 (lm_head)
ln_f = nn.LayerNorm(n_embd)
x = ln_f(x)
lm_head = nn.Linear(n_embd, vocab_size)
logits = lm_head(x) # (1, 4, 65)

print(f"\n8. 终点投射 (lm_head):")
print(f"   LayerNorm 后 shape: {x.shape}")
print(f"   lm_head(x) 投射到词表大小 shape: {logits.shape}")
print(f"   每个时间步都得到了针对 65 个候选字符的未归一化打分 (logits)！")
print("="*60)
