# FastWindowAttention

**57x faster than SDPA on 128k tokens. No OOM.**

## Benchmarks (Tesla T4)
| Sequence Length | SDPA Time | FastWindow Time | Speedup |
| :--- | :--- | :--- | :--- |
| 32k tokens | 0.23s | 0.03s | **7.8x** |
| 64k tokens | 0.73s | 0.03s | **27x** |
| 128k tokens | 2.76s | 0.05s | **55x** |
| 200k tokens | 7.54s | 0.13s | **57x** |

## Installation
Copy `fast_attention.py` into your project.

## Usage
```python
import torch
from fast_attention import FastWindowAttention

model = FastWindowAttention(hidden_size=768)
x = torch.randn(1, 128000, 768)
out, _ = model(x)
