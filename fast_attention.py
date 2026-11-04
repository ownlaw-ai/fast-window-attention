import torch
import torch.nn as nn
import torch.nn.functional as F

class FastWindowAttention(nn.Module):
    def __init__(self, hidden_size, window_size=1024, sink_size=4):
        super().__init__()
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.window_size = window_size
        self.sink_size = sink_size

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        if N <= self.window_size:
            q = q.view(B, N, 1, C).transpose(1, 2)
            k = k.view(B, N, 1, C).transpose(1, 2)
            v = v.view(B, N, 1, C).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, C)
            return self.out(out), None

        num_chunks = (N + self.window_size - 1) // self.window_size
        pad_len = num_chunks * self.window_size - N
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        q = q.view(B, num_chunks, self.window_size, C)
        k = k.view(B, num_chunks, self.window_size, C)
        v = v.view(B, num_chunks, self.window_size, C)

        k_sinks = k[:, 0, :self.sink_size, :].unsqueeze(1).expand(-1, num_chunks, -1, -1)
        v_sinks = v[:, 0, :self.sink_size, :].unsqueeze(1).expand(-1, num_chunks, -1, -1)

        k_blocks = torch.cat([k_sinks, k], dim=2)
        v_blocks = torch.cat([v_sinks, v], dim=2)

        q_blocks = q.view(B * num_chunks, 1, self.window_size, C)
        k_blocks = k_blocks.view(B * num_chunks, 1, self.window_size + self.sink_size, C)
        v_blocks = v_blocks.view(B * num_chunks, 1, self.window_size + self.sink_size, C)

        out_blocks = F.scaled_dot_product_attention(q_blocks, k_blocks, v_blocks)

        out = out_blocks.view(B, num_chunks, self.window_size, C)
        out = out.view(B, -1, C)[:, :N, :]
        return self.out(out), None
