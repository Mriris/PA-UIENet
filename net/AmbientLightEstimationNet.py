import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头注意力层（MHSA）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 将嵌入分割成多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries,keys])  # Queries 形状: (N, query_len, heads, head_dim)，Keys 形状: (N, key_len, heads, head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)),
                                  dim=3)  # attention 形状: (N, heads, query_len, key_len)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # 输出形状: (N, query_len, heads, head_dim)，然后将最后两个维度展平
        out = self.fc_out(out)
        return out
# 定义多层感知器（MLP）
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 定义Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
# 定义环境光估计流（Ambient Light Estimation Stream）网络
class AmbientLightEstimationNet(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embed_size, num_heads, num_layers, forward_expansion, dropout):
        super(AmbientLightEstimationNet, self).__init__()
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(3, embed_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.to_cls_token = nn.Identity()
        self.fc = nn.Linear(embed_size, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        mask = None
        for transformer in self.transformer_blocks:
            x = transformer(x, x, x, mask)
        x = self.to_cls_token(x[:, 0])
        out = self.fc(x)
        return out

# 实例化并测试网络
model = AmbientLightEstimationNet(
    image_size=256,
    patch_size=16,
    num_channels=3,
    embed_size=512,
    num_heads=8,
    num_layers=6,
    forward_expansion=4,
    dropout=0.1
)

# 假设输入是一张3通道的RGB图像，尺寸为256x256
input_tensor = torch.randn(1, 3, 256, 256)

# 前向传播
output_tensor = model(input_tensor)

# 打印输出
print(output_tensor)
