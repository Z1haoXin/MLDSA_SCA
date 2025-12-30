import re
import numpy as np

log_file = "search_out.log"

times = []
hit_rates = []

with open(log_file, "r") as f:
    content = f.read()

matches = re.findall(r'No\.\d+ .*?hit rate = (\d+).*?time = ([\d\.]+)s', content)

for hit, t in matches:
    hit_rates.append(int(hit))
    times.append(float(t))

if not times:
    print("no data")
else:
    times_np = np.array(times)
    hits_np = np.array(hit_rates)

    print(f"total: {len(times)}\n")

    print("=== Time (s) ===")
    print(f"min: {times_np.min():.6f}")
    print(f"max: {times_np.max():.6f}")
    print(f"median: {np.median(times_np):.6f}")
    print(f"ave: {times_np.mean():.6f}\n")

    print("=== Hit Rate ===")
    print(f"min: {hits_np.min()}")
    print(f"max: {hits_np.max()}")
    print(f"median: {np.median(hits_np)}")
    print(f"ave: {hits_np.mean():.2f}")
