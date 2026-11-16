FastWindowAttention
57x faster than SDPA on 128k tokens. No OOM. Scales to arbitrary context lengths.

Что это?
FastWindowAttention — это оптимизированная реализация Sliding Window Attention с attention sinks. Она позволяет обрабатывать последовательности неограниченной длины (теоретически до бесконечности) на потребительских GPU (Tesla T4), в то время как стандартный nn.MultiheadAttention падает с ошибкой Out of Memory (OOM) уже на 32k токенах.

Код написан на PyTorch и использует F.scaled_dot_product_attention для максимальной скорости.

Benchmarks (Tesla T4)
Sequence Length	SDPA Time	FastWindow Time	Speedup
32k tokens	0.23s	0.03s	7.8x
64k tokens	0.73s	0.03s	27x
128k tokens	2.76s	0.05s	55x
200k tokens	7.54s	0.13s	57x
Примечание: Стандартный SDPA (Scaled Dot Product Attention) не может обработать 128k+ токенов на T4 из-за нехватки памяти. FastWindowAttention справляется.

Установка
Скопируйте fast_attention.py в свой проект.

Использование
python
import torch
from fast_attention import FastWindowAttention

model = FastWindowAttention(hidden_size=768)
x = torch.randn(1, 128000, 768)
out, _ = model(x)
Как это работает?
Вместо того чтобы вычислять внимание для всех токенов (что дает квадратичную сложность O(n²) и жрет память), FastWindowAttention:

Разбивает последовательность на скользящие окна (chunks).

Для каждого окна считает внимание только к последним N токенам (окно) + первым K токенам (attention sinks).

Это дает линейную сложность O(n) и позволяет обрабатывать последовательности любой длины без OOM.

Статус
Ищу пилотный проект или сотрудничество. Открыт к обсуждению лицензирования. Если у вас есть задача, где стандартный Attention не справляется с длинным контекстом — напишите.

Контакты
Telegram: @TGVlP1
