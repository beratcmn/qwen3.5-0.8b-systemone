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

## CUDA graphs and fused linear attention

Measured on 2026-10-07 on the same RTX 3060 with the same command plus
`--questions 1 3 5 10 16 50`. "Before" is the table above; the other columns
include `flash-linear-attention` running on `triton-windows`.

| Questions | Before median | Graphs off | Graphs on | Speedup vs before | Graphs-on questions/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 173.32 ms | 138.60 ms | 20.85 ms | 8.3× | 47.96 |
| 3 | — | 144.61 ms | 42.71 ms | — | 70.23 |
| 5 | 169.46 ms | 137.57 ms | 68.19 ms | 2.5× | 73.32 |
| 10 | 251.18 ms | 135.67 ms | 122.90 ms | 2.0× | 81.37 |
| 16 | — | 195.18 ms | 181.89 ms | — | 87.97 |
| 50 | 1003.06 ms | 711.37 ms | 669.13 ms | 1.5× | 74.72 |

A profile of the 1-question request showed about 37 ms of GPU work inside
170 ms of wall time and about 3,000 kernel launches: the eager forward was
launch-bound. The single-pass text-only path now replays a CUDA graph captured
per exact batch size and 16-token width bucket, which removes that overhead.
Graphs are captured lazily, so the first request of each new shape pays a
one-time capture cost. `SYSTEMONE_CUDA_GRAPHS=0` restores the eager path.

Beyond a few questions the forward is GPU-bound, and graphs help less. There,
`chunk_gated_delta_rule` from `flash-linear-attention` replaces the fp32
reference implementation and cuts GPU time by about 35% at batch 16. Qwen3.5
still uses the reference `causal_conv1d_fn` because `causal-conv1d` publishes
no Windows wheels; it is about 10% of GPU time at batch 16. Images, long
prompts, and multi-batch requests still use the eager shared-prefix path.
