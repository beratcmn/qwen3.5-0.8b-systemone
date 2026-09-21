from __future__ import annotations

import argparse
import base64
import ctypes
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from ctypes import wintypes
from typing import Any

from PIL import Image, ImageDraw, ImageGrab, ImageOps

SYSTEMONE_URL = os.getenv(
    "SYSTEMONE_URL", "http://127.0.0.1:8000/v1/systemone"
).rstrip("/")
TELEMETRY_URL = os.getenv(
    "DINO_TELEMETRY_URL",
    SYSTEMONE_URL.removesuffix("/v1/systemone") + "/v1/dino/telemetry",
)
DASHBOARD_URL = TELEMETRY_URL.removesuffix("/v1/dino/telemetry") + "/demos/dino"

VK_SPACE = 0x20
VK_DOWN = 0x28
KEYEVENTF_KEYUP = 0x0002
GA_ROOT = 2
user32 = ctypes.windll.user32
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]


def post_json(url: str, payload: dict[str, Any], timeout: float = 180) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error.reason}") from error


def cursor_position() -> tuple[int, int]:
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError("could not read the cursor position")
    return point.x, point.y


def calibrate() -> tuple[int, int, int, int]:
    print("Move the cursor to the TOP-LEFT of the Dino playfield, then press Enter.")
    input()
    left, top = cursor_position()
    print(f"Top-left: {left}, {top}")
    print("Move the cursor to the BOTTOM-RIGHT of the playfield, then press Enter.")
    input()
    right, bottom = cursor_position()
    if right - left < 200 or bottom - top < 60:
        raise RuntimeError("the selected playfield is too small")
    return left, top, right, bottom


def root_window_at(bbox: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bbox
    point = wintypes.POINT((left + right) // 2, (top + bottom) // 2)
    window = user32.WindowFromPoint(point)
    root = user32.GetAncestor(window, GA_ROOT)
    if not root:
        raise RuntimeError("could not find the Chrome window under the playfield")
    return root


def window_title(window: int) -> str:
    length = user32.GetWindowTextLengthW(window)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(window, buffer, len(buffer))
    return buffer.value or "selected window"


def window_is_focused(window: int) -> bool:
    foreground = user32.GetAncestor(user32.GetForegroundWindow(), GA_ROOT)
    return foreground == window


def key_down(key: int) -> None:
    user32.keybd_event(key, 0, 0, 0)


def key_up(key: int) -> None:
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)


def tap(key: int, duration: float = 0.035) -> None:
    key_down(key)
    time.sleep(duration)
    key_up(key)


def capture_frame(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> Image.Image:
    source = ImageGrab.grab(bbox=bbox, all_screens=True)
    try:
        fitted = ImageOps.fit(
            source.convert("L"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
        return fitted.point(lambda value: 255 if value > 180 else 0).convert("RGB")
    finally:
        source.close()


def motion_strip(
    bbox: tuple[int, int, int, int], width: int, height: int, frame_gap: float
) -> Image.Image:
    previous = capture_frame(bbox, width, height)
    time.sleep(frame_gap)
    current = capture_frame(bbox, width, height)
    try:
        strip = Image.new("RGB", (width * 2, height), "white")
        strip.paste(previous, (0, 0))
        strip.paste(current, (width, 0))
        ImageDraw.Draw(strip).line((width, 0, width, height), fill=(128, 128, 128), width=1)
        return strip
    finally:
        previous.close()
        current.close()


def observation(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    frame_count: int,
    frame_gap: float,
) -> Image.Image:
    if frame_count == 1:
        return capture_frame(bbox, width, height)
    return motion_strip(bbox, width, height, frame_gap)


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode()


def decide(
    frame_base64: str, frame_count: int = 1
) -> tuple[str, dict[str, float], float, float]:
    visual_description = (
        "The attached image is the newest game view."
        if frame_count == 1
        else (
            "The attached strip contains two consecutive views: the older frame "
            "is left and the newest frame is right."
        )
    )
    payload = {
        "model": "qwen3.5-0.8b-systemone",
        "state": (
            f"Play Chrome Dino using pixels only. {visual_description} "
            "The action reaches the game about 0.8 seconds after capture, so act early."
        ),
        "attachments": [
            {
                "id": "motion_strip",
                "media_type": "image/png",
                "data_base64": frame_base64,
            }
        ],
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "Choose the safest action for the newest frame, accounting for the "
                    "control delay."
                    + (
                        " Use the older frame only to judge motion and closing distance."
                        if frame_count == 2
                        else ""
                    )
                ),
                "criteria": {
                    "wait": "The game is running and no obstacle requires action yet.",
                    "jump": "A cactus, ground obstacle, or low flyer is close enough to jump now.",
                    "duck": "A flying obstacle at head height is close enough to duck under now.",
                    "restart": "The dinosaur has crashed or the game is waiting to start.",
                },
            }
        },
    }
    started = time.perf_counter()
    result = post_json(SYSTEMONE_URL, payload)
    latency_ms = (time.perf_counter() - started) * 1000
    answer = result["answers"]["action"]
    return answer["choice"], answer["probabilities"], answer["confidence"], latency_ms


def telemetry(
    session_id: str,
    sequence: int,
    status: str,
    **values: Any,
) -> None:
    payload = {
        "session_id": session_id,
        "sequence": sequence,
        "captured_at": time.time(),
        "status": status,
        **values,
    }
    post_json(TELEMETRY_URL, payload, timeout=10)


def execute(
    model_action: str,
    probabilities: dict[str, float],
    threshold: float,
    jump_cooldown: float,
    last_jump: float,
    duck_held: bool,
) -> tuple[str, float, bool]:
    action = model_action if probabilities[model_action] >= threshold else "wait"
    now = time.monotonic()
    if action == "jump" and now - last_jump < jump_cooldown:
        action = "wait"

    if action != "duck" and duck_held:
        key_up(VK_DOWN)
        duck_held = False
    if action == "duck" and not duck_held:
        key_down(VK_DOWN)
        duck_held = True
    elif action == "jump":
        tap(VK_SPACE)
        last_jump = now
    elif action == "restart":
        tap(VK_SPACE)
        last_jump = now
    return action, last_jump, duck_held


def main() -> None:
    if os.name != "nt":
        raise SystemExit("This visual controller currently requires Windows.")

    parser = argparse.ArgumentParser(
        description="Play the real Chrome Dino game using visual System One decisions."
    )
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--frames", type=int, choices=(1, 2), default=1)
    parser.add_argument("--frame-gap", type=float, default=0.06)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--jump-cooldown", type=float, default=0.45)
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else calibrate()
    target = root_window_at(bbox)
    session_id = uuid.uuid4().hex[:12]
    sequence = 0
    duck_held = False
    last_jump = 0.0
    paused = False
    failed = False

    print(f"Playfield: {bbox}")
    print(f"Target: {window_title(target)}")
    print(f"Dashboard: {DASHBOARD_URL}")
    print("Keep the Dino window focused. Press Ctrl+C here to stop.")
    telemetry(session_id, sequence, "calibrating")
    print("Warming the visual model before the game starts...")
    warmup_image = observation(
        bbox, args.width, args.height, args.frames, args.frame_gap
    )
    try:
        warmup_data = encode_png(warmup_image)
    finally:
        warmup_image.close()
    _, _, _, warmup_latency = decide(warmup_data, args.frames)
    print(f"Visual model ready in {warmup_latency:.0f}ms.")
    user32.SetForegroundWindow(target)
    time.sleep(0.8)
    if not args.no_start:
        tap(VK_SPACE)
        time.sleep(0.7)

    try:
        while True:
            if not window_is_focused(target):
                if duck_held:
                    key_up(VK_DOWN)
                    duck_held = False
                if not paused:
                    telemetry(session_id, sequence, "paused")
                    print("Paused: focus the Dino window to continue.")
                    paused = True
                time.sleep(0.25)
                continue
            if paused:
                telemetry(session_id, sequence, "running")
                print("Dino window focused; continuing.")
                paused = False

            image = observation(
                bbox, args.width, args.height, args.frames, args.frame_gap
            )
            try:
                encoded = encode_png(image)
            finally:
                image.close()
            model_action, probabilities, confidence, latency_ms = decide(
                encoded, args.frames
            )
            executed, last_jump, duck_held = execute(
                model_action,
                probabilities,
                args.threshold,
                args.jump_cooldown,
                last_jump,
                duck_held,
            )
            sequence += 1
            telemetry(
                session_id,
                sequence,
                "running",
                frame_count=args.frames,
                frame_base64=encoded,
                model_action=model_action,
                executed_action=executed,
                probabilities=probabilities,
                confidence=confidence,
                latency_ms=latency_ms,
            )
            print(
                f"#{sequence:04d} model={model_action:<7} sent={executed:<7} "
                f"p={probabilities[model_action]:.2f} latency={latency_ms:.0f}ms"
            )
            time.sleep(max(0, args.interval))
    except KeyboardInterrupt:
        print("\nStopping Dino controller.")
    except Exception as error:
        failed = True
        try:
            telemetry(session_id, sequence, "error", error=str(error))
        except RuntimeError:
            pass
        raise
    finally:
        if duck_held:
            key_up(VK_DOWN)
        if not failed:
            try:
                telemetry(session_id, sequence, "stopped")
            except RuntimeError:
                pass


if __name__ == "__main__":
    main()
