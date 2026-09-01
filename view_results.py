import json

with open("predictions.json") as f:
    results = json.load(f)

for r in results:
    label = "AI-generated" if r["pred"] >= 0.5 else "Real"
    print(f"{r['image_path']}: {r['pred']:.3f} ({label})")