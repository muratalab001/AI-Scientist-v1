import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

folders = os.listdir("./")
final_results = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)

for run in final_results.keys():
    traces = final_results[run]["SO(N)"]["traces"]
    determinants = final_results[run]["SO(N)"]["determinants"]
    plt.figure(figsize=(8, 4))
    plt.hist(traces, bins=30, color="blue", alpha=0.7)
    plt.title(f"Trace distribution for {run}")
    plt.xlabel("Trace")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"trace_hist_{run}.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(determinants, bins=30, color="green", alpha=0.7)
    plt.title(f"Determinant distribution for {run}")
    plt.xlabel("Determinant")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"determinant_hist_{run}.png")
    plt.close()
