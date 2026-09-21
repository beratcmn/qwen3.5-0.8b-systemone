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
