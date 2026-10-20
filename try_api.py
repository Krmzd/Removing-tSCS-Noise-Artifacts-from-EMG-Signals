import json
import urllib.request

import pandas as pd

# Take the first 2000 samples of one real recording
df = pd.read_csv("data/MJ/BB_tSCS_before_BLT.csv")
signal = df["1 L BB"].fillna(0).values[:2000].tolist()

request = urllib.request.Request(
    "http://127.0.0.1:8000/denoise",
    data=json.dumps({"signal": signal}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request) as response:
    result = json.loads(response.read())

print("Sent", len(signal), "samples, got back", len(result["cleaned"]))
print("First 5 noisy:  ", [round(x, 4) for x in signal[:5]])
print("First 5 cleaned:", [round(x, 4) for x in result["cleaned"][:5]])
