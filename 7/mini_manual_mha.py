import torch
import torch.nn.functional as F

# 设定极简数字
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # 第 0 个字
    [0.0, 2.0, 0.0, 1.0]   # 第 1 个字
]) # (T=2, C=4)

print("="*50)
print("1. 输入矩阵 X (2个字, 4维特征):")
print(X)

# Head 0 的参数 (4x2)
Wq0 = torch.tensor([[1.,0.], [0.,1.], [1.,0.], [0.,1.]])
Wk0 = torch.tensor([[1.,0.], [0.,1.], [0.,0.], [1.,1.]])
Wv0 = torch.tensor([[10.,0.], [0.,20.], [0.,0.], [0.,0.]])

# 算 Head 0 的 Q, K, V
Q0 = X @ Wq0 # (2, 2)
K0 = X @ Wk0 # (2, 2)
V0 = X @ Wv0 # (2, 2)

print("\n2. Head 0 内部计算出的 Q, K, V:")
print(f"Q0:\n{Q0}\nK0:\n{K0}\nV0:\n{V0}")

# 算打分与掩码
scores = Q0 @ K0.T # (2, 2)
tril = torch.tril(torch.ones(2, 2))
masked_scores = scores.masked_fill(tril == 0, float('-inf'))
wei0 = F.softmax(masked_scores, dim=-1)

print("\n3. Head 0 注意力权重表 wei (不能看未来):")
print(wei0)

out0 = wei0 @ V0 # (2, 2)
print("\n4. Head 0 最终输出 out0 (wei @ V):")
print(out0)

# 假设 Head 1 算出的输出是 out1
out1 = torch.tensor([
    [5.0, 1.0],
    [2.0, 8.0]
])
print("\n5. Head 1 算出的输出 out1:")
print(out1)

# 多头拼接
multi_out = torch.cat([out0, out1], dim=-1) # (2, 4)
print("\n6. 两头并排拼接 torch.cat([out0, out1], dim=-1):")
print(multi_out)
print(f"形状恢复到了: {multi_out.shape}")
print("="*50)
