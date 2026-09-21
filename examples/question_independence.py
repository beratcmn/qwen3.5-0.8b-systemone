"""Compare one question alone with the same question inside a mixed batch."""

from _client import post

state = "The production API returns HTTP 503 for all users. A rollback is available."
target = {
    "type": "choice",
    "instructions": "Which team should own this incident?",
    "criteria": {
        "engineering": "Application failures and outages",
        "billing": "Charges, invoices, and refunds",
        "sales": "Pricing and new purchases",
    },
}
base = {"model": "qwen3.5-0.8b-systemone", "state": state}

alone = post({**base, "questions": {"owner": target}})["answers"]["owner"]
batched = post(
    {
        **base,
        "questions": {
            "owner": target,
            "tone": {
                "type": "choice",
                "instructions": "What is the tone of the incident report?",
                "criteria": {"calm": None, "urgent": None, "angry": None},
            },
            "rollback": {
                "type": "noul",
                "instructions": "Is a rollback available?",
            },
        },
    }
)["answers"]["owner"]

deltas = {
    option: abs(alone["probabilities"][option] - batched["probabilities"][option])
    for option in alone["probabilities"]
}
print("alone:  ", alone)
print("batched:", batched)
print("absolute probability deltas:", deltas)

if alone["choice"] != batched["choice"] or max(deltas.values()) > 0.03:
    raise SystemExit("Question independence check failed")
print("PASS: winner is unchanged and BF16 probability drift is within 0.03")
