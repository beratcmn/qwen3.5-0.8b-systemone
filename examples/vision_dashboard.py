"""Generate a synthetic outage dashboard and send it through Qwen's vision path."""

import base64
import io

from PIL import Image, ImageDraw

from _client import post, show

image = Image.new("RGB", (640, 360), "#111827")
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((35, 35, 605, 325), radius=20, fill="#1f2937")
draw.text((70, 65), "PAYMENTS SERVICE", fill="white", font_size=32)
draw.rounded_rectangle((70, 130, 570, 280), radius=16, fill="#b91c1c")
draw.text((165, 165), "STATUS: DOWN", fill="white", font_size=38)
draw.text((190, 225), "ERROR RATE 92%", fill="white", font_size=26)
buffer = io.BytesIO()
image.save(buffer, format="PNG")

payload = {
    "model": "qwen3.5-0.8b-systemone",
    "state": "Inspect attachment `payments_dashboard` and assess the service shown in it.",
    "attachments": [
        {
            "id": "payments_dashboard",
            "media_type": "image/png",
            "data_base64": base64.b64encode(buffer.getvalue()).decode(),
        }
    ],
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What status does the dashboard show?",
            "criteria": {
                "healthy": "The service is operating normally",
                "degraded": "The service has partial problems",
                "outage": "The service is down or almost completely failing",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the displayed service condition?",
            "criteria": ["Normal", "Degraded", "Critical outage"],
        },
        "alert_oncall": {
            "type": "noul",
            "instructions": "Should the on-call engineer be alerted immediately?",
        },
    },
}

show(post(payload))
