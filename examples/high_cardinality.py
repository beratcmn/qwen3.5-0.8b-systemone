"""Exercise the full 255-option Choice contract and inspect its semantic quality."""

from _client import post

target = 173
criteria = {
    f"product_{index:03d}": f"Catalog product with numeric identifier {index:03d}"
    for index in range(255)
}
result = post(
    {
        "model": "qwen3.5-0.8b-systemone",
        "state": {
            "requested_product": f"product_{target:03d}",
            "numeric_identifier": f"{target:03d}",
        },
        "questions": {
            "product": {
                "type": "choice",
                "instructions": "Select the exact product named in `requested_product`.",
                "criteria": criteria,
            }
        },
    }
)
answer = result["answers"]["product"]
top_five = sorted(
    answer["probabilities"].items(), key=lambda item: item[1], reverse=True
)[:5]

print("expected:", f"product_{target:03d}")
print("selected:", answer["choice"])
print("candidate count:", len(answer["probabilities"]))
print("probability sum:", sum(answer["probabilities"].values()))
print("top five:")
for option, probability in top_five:
    print(f"  {option}: {probability:.6f}")
