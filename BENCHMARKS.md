# Benchmarks

Preliminary local result from 2026-09-21 on an NVIDIA GeForce RTX 3060 12 GB, PyTorch 2.9.1 CUDA 12.8, Transformers 5.17.0, BF16, and the standard Windows fallback kernels:

```powershell
uv run python scripts/benchmark.py --questions 4 --state-repetitions 4 --runs 2
```

| Mode | Batch | Median | p95 | Questions/s | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent request per question | — | 1393.7 ms | 1401.9 ms | 2.87 | 1723.7 MiB |
| Shared prefill, sequential branches | 1 | 873.4 ms | 881.9 ms | 4.58 | 1723.7 MiB |
| Shared prefill, parallel branches | 4 | 359.1 ms | 363.0 ms | 11.14 | 1840.7 MiB |

Model loading and the one warmup run per mode are excluded. These two-run figures verify the execution design; use the benchmark's default five runs, and the intended state/question sizes, before making production capacity decisions.

## Matched short-text workload

The HTTP benchmark uses one short customer message and cycles through a choice,
score, and noul question. Each result below is from 3 warmups and 12 measured
runs with `SYSTEMONE_BATCH_SIZE=16` on the same RTX 3060 system.

```powershell
uv run python scripts/benchmark_short.py --runs 12
```

| Questions | Before median | Optimized median | Improvement | Optimized questions/s | Input tokens before → after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 343.67 ms | 173.32 ms | 49.6% | 5.77 | 166 → 107 |
| 5 | 351.35 ms | 169.46 ms | 51.8% | 29.51 | 547 → 335 |
| 10 | 439.22 ms | 251.18 ms | 42.8% | 39.81 | 979 → 599 |
| 50 | 1540.87 ms | 1003.06 ms | 34.9% | 49.85 | 4612 → 2795 |

The implementation now:

- encodes criteria in a compact label table and caches repeated question templates;
- expands a shared cache directly instead of deep-copying every tensor first;
- evaluates a text-only request in one model pass when all questions fit one batch
  and the longest complete prompt is at most 128 tokens;
- retains shared-prefix inference for images, long prompts, and multi-batch requests.

The component profile showed why the single-pass path matters. Before these
changes, one short question spent about 186 ms in prefix inference and 172 ms in
suffix inference. Template preparation and cache expansion took about 2 ms and
10 ms. For 50 questions, suffix inference consumed about 1232 ms of the 1500 ms
profiled total. Compact prompts reduced the longest suffix from 110 to 64 tokens,
and direct cache expansion reduced cache work from about 9.7 ms to 3.0 ms per
batch.

Qwen3.5 still reports the reference PyTorch implementations of
`causal_conv1d_fn` and `chunk_gated_delta_rule` on this Windows installation.
The current PyPI releases of `causal-conv1d` and `flash-linear-attention` do not
publish Windows wheels, so this benchmark does not claim fused-kernel speed.
Use a supported Linux/CUDA environment for those kernels, then rerun the same
HTTP benchmark to measure the actual gain on that machine.
