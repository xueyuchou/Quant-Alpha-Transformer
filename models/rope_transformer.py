import torch
import torch.nn as nn
import math

class RoPEAttention(nn.Module):
    def __init__(self, embed_size, seq_len=20):
        super().__init__()
        self.embed_size = embed_size
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)

        # RoPE ：预先计算好 20 天的旋转角度
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, embed_size, 2).float() / embed_size))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.sin(freqs))

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = (Q * self.cos) + (self.rotate_half(Q) * self.sin)
        K = (K * self.cos) + (self.rotate_half(K) * self.sin)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.embed_size)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

class Transformerblock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attention = RoPEAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        forwarded = self.ff(x)
        out = self.norm2(x + forwarded)
        return out

class QuantTransformer(nn.Module):
    def __init__(self, embed_size, output_size=1, seq_len=20):
        super().__init__()
        self.block = Transformerblock(embed_size)
        self.fc = nn.Linear(embed_size, output_size)

    def forward(self, x):
        out = self.block(x)
        final_day_state = out[:, -1, :]
        prediction = self.fc(final_day_state).squeeze(-1)
        return prediction