# qwen3.5-0.8b-systemone

A local Jev-style decision API powered by `Qwen/Qwen3.5-0.8B`. One request can ask any mixture of Choice, Score, and Noul questions. The model scores constrained candidates; application code constructs the JSON response.

## Setup

Requirements: Windows, Python 3.12, `uv`, an NVIDIA GPU, and a recent NVIDIA driver. The default environment uses PyTorch's CUDA 12.8 wheels.

```powershell
uv sync
uv run qwen3-5-0-8b-systemone
```

The first launch downloads the pinned Qwen checkpoint, verifies 255 single-token internal labels, loads the model on CUDA, and starts the API at `http://127.0.0.1:8000`. OpenAPI documentation is available at `/docs`; `/healthz` checks the HTTP process and `/readyz` reports whether the model is loaded.

## API

```powershell
$body = @{
  model = "qwen3.5-0.8b-systemone"
  state = @{
    message = "Checkout fails for every customer after deployment. There is no workaround."
    environment = "production"
  }
  questions = @{
    department = @{
      type = "choice"
      instructions = "Which team should handle this incident?"
      criteria = @{
        engineering = "Application failures, outages, and deployment problems"
        billing = "Invoices, charges, and refunds"
        sales = "Pricing, purchasing, and upgrades"
      }
    }
    severity = @{
      type = "score"
      instructions = "How severe is the disruption?"
      criteria = @("No impact", "Partial disruption", "Critical without a workaround")
    }
    needs_oncall = @{
      type = "noul"
      instructions = "Does this need immediate on-call attention?"
    }
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://127.0.0.1:8000/v1/systemone `
  -Method Post -ContentType application/json -Body $body
```

Choice responses contain the selected public option and every option probability. Score responses contain an expected rubric index, legend, probabilities, and confidence. Noul returns the probability of `true`. Choice and Score confidence is normalized entropy: `1 - H(p) / log(n)`. `usage.output_tokens` is zero because the engine reads candidate logits without sampling tokens.

Images can be supplied as up to four base64-encoded PNG, JPEG, or static WebP attachments:

```json
{
  "attachments": [
    {"id": "screen", "media_type": "image/png", "data_base64": "..."}
  ]
}
```

The question can refer to an attachment by its `id`. Images are processed by Qwen's native vision path.

## Development

```powershell
uv run pytest
$env:SYSTEMONE_INTEGRATION = "1"; uv run pytest tests/test_integration.py
uv run python scripts/benchmark.py --questions 4 --runs 5
```

The server accepts one active inference request and returns HTTP 529 while the GPU is busy. Set `SYSTEMONE_BATCH_SIZE` to tune the number of question branches evaluated together. The public model alias remains fixed; `SYSTEMONE_MODEL_REVISION` may override the pinned checkpoint revision for compatibility testing.
