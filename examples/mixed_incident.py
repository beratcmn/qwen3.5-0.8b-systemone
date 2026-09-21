"""Exercise Choice, Score, and Noul together in one request."""

from _client import post, show

payload = {
    "model": "qwen3.5-0.8b-systemone",
    "state": {
        "message": "Checkout fails for every customer after the deployment. There is no workaround.",
        "environment": "production",
        "affected_customers_percent": 100,
    },
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this incident?",
            "criteria": {
                "engineering": "Application failures, outages, and deployment problems",
                "billing": "Invoices, charges, and refund requests",
                "sales": "Pricing, purchasing, and account upgrades",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the service disruption?",
            "criteria": [
                "No functional impact",
                "Partial disruption with a usable workaround",
                "Critical disruption without a workaround",
            ],
        },
        "needs_oncall": {
            "type": "noul",
            "instructions": "Does this incident require immediate on-call attention?",
            "criteria": {
                "true": "An active production failure requires urgent intervention",
                "false": "The issue can wait for normal support handling",
            },
        },
    },
}

show(post(payload))
