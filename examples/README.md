# Examples

Start the API in one terminal:

```powershell
uv run qwen3-5-0-8b-systemone
```

Open the interactive Connect Four demo at [http://127.0.0.1:8000/demos/connect-four](http://127.0.0.1:8000/demos/connect-four). You play coral; Qwen plays yellow and shows its complete move distribution after every turn.

### Chrome Dino visual controller

Open `chrome://dino` in Chrome and keep the game visible. In another terminal,
start the visual controller:

```powershell
uv run python examples/dino_player.py
```

For this demo, restart the server with Qwen's image-resize floor lowered so the
small game strip is not enlarged:

```powershell
$env:SYSTEMONE_IMAGE_MIN_PIXELS = "4096"
uv run qwen3-5-0-8b-systemone
```

The controller asks for the top-left and bottom-right corners of the visible
playfield. Move the cursor to each corner and press Enter in the terminal. It
then captures a 128×32 frame, asks the model to choose `wait`, `jump`, `duck`,
or `restart`, and sends the guarded keyboard action to the focused Chrome
window. Pass `--frames 2` to use a two-frame 256×32 motion strip at higher
latency. The first visual request warms the model before the controller starts
the game.

Open [http://127.0.0.1:8000/demos/dino](http://127.0.0.1:8000/demos/dino) to
watch the exact visual input, action probabilities, executed actions, latency,
and recent decision history. The controller uses screenshots and keyboard input
only; it does not inspect the DOM, canvas, or game variables.

After calibrating once, reuse the printed rectangle directly:

```powershell
uv run python examples/dino_player.py --bbox LEFT TOP RIGHT BOTTOM
```

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
