import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# 读取数据
with open('7/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 超轻量配置：用于快速训练并观察注意力
block_size = 16
batch_size = 64
n_embd = 64
n_head = 4
n_layers = 2
dropout = 0.0
lr = 1e-3

def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.last_wei = None # 保存最近一次注意力权重用于可视化

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach() # 记录下来
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# 实例化并训练 400 步
print(f"使用设备: {device}")
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("正在快速训练小模型 (400 步)...")
for step in range(401):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

# 测试句子可视化注意力
test_str = "First Citizen:\n"  # 恰好 15 个字符
test_tokens = torch.tensor([encode(test_str)], dtype=torch.long, device=device)

# 进行前向传播，捕获注意力
model.eval()
with torch.no_grad():
    model(test_tokens)

# 拿第 1 层 Block 的各个 Head 的注意力权重
block0 = model.blocks[0]
heads = block0.sa.heads

print("\n" + "="*70)
print(f"【测试句子】: {repr(test_str)} (长度 {len(test_str)})")
print("="*70)

# 展示各个 Head 在看什么
# 打印一个格式化的字符注意力矩阵
# 选 Head 0 来展示完整的 T x T 矩阵
wei_h0 = heads[0].last_wei[0].cpu().numpy() # (T, T)
chars_list = [c if c != '\n' else '\\n' for c in test_str]

print(f"\n>>> Block 0 - Head 0 注意力权重矩阵 wei (数值为 % 占比):")
header = "      " + " ".join([f"{c:>4}" for c in chars_list])
print(header)
print("     " + "-" * (len(header) - 5))

for i, row_char in enumerate(chars_list):
    row_vals = " ".join([f"{int(wei_h0[i, j]*100):4d}" if j <= i else "   ." for j in range(len(chars_list))])
    print(f"{row_char:>4} |{row_vals}")

print("\n>>> 分析每个头（Head）独特的关注偏好：")
for h_idx in range(n_head):
    wei_h = heads[h_idx].last_wei[0].cpu().numpy()
    # 观察最后一个字符（冒号后面的换行符）到底在看谁
    last_char_weights = wei_h[-1]
    top3_indices = last_char_weights.argsort()[::-1][:3]
    top3_str = ", ".join([f"'{chars_list[idx]}'(位置{idx}, 占比{last_char_weights[idx]*100:.1f}%)" for idx in top3_indices])
    print(f"  [Head {h_idx}] 当预测 '\\n' 的下一个字符时，注意力最高的前 3 个字: {top3_str}")
