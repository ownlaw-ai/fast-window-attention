import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Твой класс
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

# --- ТЕСТ (ЧЕСТНЫЙ) ---
device = torch.device("cuda")
seq_len = 128000 
hidden_size = 512
x = torch.randn(1, seq_len, hidden_size, device=device, dtype=torch.float16)

print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
print(f"Тестируем на {seq_len} токенах...")

# 1. ТЕСТ СТАНДАРТНОГО СЛОЯ (полный nn.MultiheadAttention)
print("\n--- nn.MultiheadAttention (полный слой) ---")
try:
    model_std = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True).to(device).half()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    _ = model_std(x, x, x)
    torch.cuda.synchronize()
    time_std = time.time() - start
    mem_std = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Время: {time_std:.4f} сек | Память: {mem_std:.2f} ГБ")
except torch.cuda.OutOfMemoryError:
    print("💥 OOM: Стандартный слой упал (как и должно быть)!")
    time_std = None

# 2. ТЕСТ ТВОЕГО СЛОЯ
print("\n--- FastWindowAttention (мой слой) ---")
try:
    model_my = FastWindowAttention(hidden_size).to(device).half()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    _ = model_my(x)
    torch.cuda.synchronize()
    time_my = time.time() - start
    mem_my = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Время: {time_my:.4f} сек | Память: {mem_my:.2f} ГБ")
except torch.cuda.OutOfMemoryError:
    print("💥 OOM: Мой слой тоже упал (значит, тест нечестный)")

# 3. ИТОГ
print("\n" + "="*40)
print("FINAL RESULTS (для Reddit):")
print("="*40)
if time_std and time_my:
    print(f"Standard: {time_std:.4f}s | {mem_std:.2f}GB")
    print(f"Mine: {time_my:.4f}s | {mem_my:.2f}GB")
    print(f"Speedup: {time_std/time_my:.2f}x")
elif time_std is None and time_my:
    print("Standard OOM'd. Mine survived.")
    print(f"Mine: {time_my:.4f}s | {mem_my:.2f}GB")
