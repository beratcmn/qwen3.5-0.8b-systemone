# Examples

Start the API in one terminal:

```powershell
uv run qwen3-5-0-8b-systemone
```

Open the interactive Connect Four demo at [http://127.0.0.1:8000/demos/connect-four](http://127.0.0.1:8000/demos/connect-four). You play coral; Qwen plays yellow and shows its complete move distribution after every turn.

Run examples from the repository root in another terminal:

```powershell
uv run python examples/mixed_incident.py
uv run python examples/turkish_triage.py
uv run python examples/vision_dashboard.py
uv run python examples/question_independence.py
uv run python examples/high_cardinality.py
```

Each example validates the response shape, probability sums, confidence range, and derived Score value before displaying results. `question_independence.py` also checks that an unrelated batch does not materially change the target answer. `high_cardinality.py` exercises all 255 options and reports semantic quality without assuming the base model will choose correctly.

Set `SYSTEMONE_URL` to test another server:

```powershell
$env:SYSTEMONE_URL = "http://other-host:8000/v1/systemone"
uv run python examples/mixed_incident.py
```
