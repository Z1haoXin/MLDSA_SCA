#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt

SIGMAS = [0.2, 0.4, 0.6, 0.8]
NUM_RUNS = 10
MAX_COEFF = 1024

plt.rcParams.update({
    "font.size":10,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "lines.linewidth": 1.8
})

def read_log(filename):
    trace_nums, recovered = [], []
    pattern = re.compile(r"trace_num=(\d+).*recovered=(\d+)")
    with open(filename) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                trace_nums.append(int(m.group(1)))
                recovered.append(int(m.group(2)))
    if not trace_nums:
        raise RuntimeError(f"No valid data in {filename}")
    return np.array(trace_nums), np.array(recovered)

def build_runs(log_files):
    runs = []
    max_trace_local = 0
    for fname in log_files:
        trace, rec = read_log(fname)
        runs.append((trace, rec))
        if len(trace) > 0:
            max_trace_local = max(max_trace_local, trace[-1])
    return runs, max_trace_local

def compute_avg_trc(runs, max_coeff):
    avg_trc = np.zeros(max_coeff, dtype=np.float64)
    first1024_list = []
    for N in range(1, max_coeff+1):
        trcs = []
        for trace, rec in runs:
            idx = np.where(rec >= N)[0]
            trc_needed = trace[idx[0]] if len(idx) else trace[-1]
            trcs.append(trc_needed)
        avg_trc[N-1] = np.mean(trcs)
        if N == MAX_COEFF:
            first1024_list = trcs
    return avg_trc, first1024_list

global_max_trace = 0
elmo_runs = {}
elmo_avg_trc = {}
elmo_first1024 = {}

for sigma in SIGMAS:
    tag = str(sigma).replace('.', 'd')
    log_files = [f"mvke_sigma{tag}_{q}.log" for q in range(NUM_RUNS)]
    runs, max_trace = build_runs(log_files)
    elmo_runs[sigma] = runs
    global_max_trace = max(global_max_trace, max_trace)
    avg_trc, first1024_list = compute_avg_trc(runs, MAX_COEFF)
    elmo_avg_trc[sigma] = avg_trc
    elmo_first1024[sigma] = np.mean(first1024_list)

cw_files = [f"mvke_{q}.log" for q in range(20)]
cw_runs, max_trace_cw = build_runs(cw_files)
global_max_trace = max(global_max_trace, max_trace_cw)
cw_avg_trc_curve, cw_first1024_list = compute_avg_trc(cw_runs, MAX_COEFF)
cw_first1024 = np.mean(cw_first1024_list)


fig, ax = plt.subplots(figsize=(8,4.6))

colors = {0.2:"tab:blue",0.4:"tab:orange",0.6:"tab:green",0.8:"tab:red"}

# ChipWhisperer
last_trc_cw = cw_avg_trc_curve[-1]
if last_trc_cw < global_max_trace:
    extend_x = np.arange(last_trc_cw, global_max_trace+1)
    extend_y = np.full(len(extend_x), MAX_COEFF)
    plot_x = np.concatenate([cw_avg_trc_curve, extend_x])
    plot_y = np.concatenate([np.arange(1, MAX_COEFF+1), extend_y])
else:
    plot_x = cw_avg_trc_curve
    plot_y = np.arange(1, MAX_COEFF+1)

ax.plot(plot_x, plot_y, linestyle='-', color='dimgray', label='ChipWhisperer')
ax.scatter(cw_first1024, MAX_COEFF, color='dimgray', s=30, zorder=6)
ax.text(cw_first1024-10, MAX_COEFF-50, f"({int(np.ceil(cw_first1024))},1024)")
ax.axvline(cw_first1024, ymax=MAX_COEFF/(MAX_COEFF+50), linestyle=':', color='dimgray', linewidth=0.9)
ax.axhline(MAX_COEFF, xmax=cw_first1024/(global_max_trace+50), linestyle=':', color='dimgray', linewidth=0.9)

# ELMO
for sigma in SIGMAS:
    curve = elmo_avg_trc[sigma]
    last_trc = curve[-1]
    if last_trc < global_max_trace:
        extend_x = np.arange(last_trc, global_max_trace+1)
        extend_y = np.full(len(extend_x), MAX_COEFF)
        plot_x = np.concatenate([curve, extend_x])
        plot_y = np.concatenate([np.arange(1, MAX_COEFF+1), extend_y])
    else:
        plot_x = curve
        plot_y = np.arange(1, MAX_COEFF+1)

    color = colors[sigma]
    ax.plot(plot_x, plot_y, color=color, linestyle='--', label=f"ELMO σ={sigma}")
    trc1024 = elmo_first1024[sigma]
    ax.scatter(trc1024, MAX_COEFF, color=color, s=30, zorder=6)
    ax.text(trc1024+5, MAX_COEFF+10, f"({int(np.ceil(trc1024))},1024)")
    ax.axvline(trc1024, ymax=MAX_COEFF/(MAX_COEFF+50), linestyle=':', color=color, linewidth=0.9)
    ax.axhline(MAX_COEFF, xmax=trc1024/(global_max_trace+50), linestyle=':', color=color, linewidth=0.9)

ax.set_xlabel(r"Average number of equations $N_{eq}$")
ax.set_ylabel(r"Recovered coefficients $N_{coef}$")
ax.set_xlim(0, global_max_trace+50)
ax.set_ylim(0, MAX_COEFF+50)
ax.grid(alpha=0.3)
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig("recovery_curves.pdf", dpi=300)
plt.show()
