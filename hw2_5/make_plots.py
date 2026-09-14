"""
Reads the CSVs/RUN_LOG produced by parts B, D, E and produces the required
figures. Run after all benchmark scripts have completed on a given card.

Usage: python make_plots.py --card rtx4090
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_part_b(card: str):
    df = pd.read_csv(f"{card}_part_b_throughput.csv")
    df = df[df["achieved_tflops"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    for precision, group in df.groupby("precision"):
        group = group.sort_values("n")
        ax.plot(group["n"], group["achieved_tflops"], marker="o", label=precision)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(f"{card}: throughput vs matrix size by precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{card}_part_b_throughput.png", dpi=150)
    print(f"Wrote {card}_part_b_throughput.png")


def load_run_log(card_filter: str = None):
    rows = []
    with open("RUN_LOG.txt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_part_d_memory(card: str):
    rows = load_run_log()
    naive_pts = [(r["config"]["seq_len"], r["result"]["peak_memory_bytes"])
                 for r in rows
                 if r["part"] == "D" and r["config"].get("impl") == "naive"
                 and "seq_len" in r["config"]
                 and not r["result"].get("oom", False)]
    if not naive_pts:
        print("No naive attention (non-OOM) points found in RUN_LOG.txt")
        return
    naive_pts.sort()
    seq_lens = np.array([p[0] for p in naive_pts], dtype=float)
    mem_gb = np.array([p[1] for p in naive_pts]) / 1e9

    # Fit peak_memory = a*S^2 + b*S + c to confirm the quadratic term from
    # data rather than asserting it, per the assignment's instruction.
    coeffs = np.polyfit(seq_lens, mem_gb, 2)
    a, b, c = coeffs
    fit_s = np.linspace(seq_lens.min(), seq_lens.max(), 200)
    fit_mem = np.polyval(coeffs, fit_s)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(seq_lens, mem_gb, label="measured (naive attention)")
    ax.plot(fit_s, fit_mem, "--",
            label=f"quadratic fit: a={a:.3e} GB/tok^2")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title(f"{card}: naive attention memory vs sequence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{card}_part_d_memory.png", dpi=150)
    print(f"Wrote {card}_part_d_memory.png; quadratic coefficient a={a:.6e}")


def plot_part_e_thermal(card: str):
    df = pd.read_csv(f"{card}_thermal_log.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["elapsed_s"], df["clocks.sm"], color="tab:blue", label="SM clock (MHz)")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("SM clock (MHz)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["elapsed_s"], df["temperature.gpu"], color="tab:red", label="Temp (C)")
    ax2.set_ylabel("Temperature (C)", color="tab:red")
    fig.suptitle(f"{card}: clock and temperature under sustained load")
    fig.tight_layout()
    fig.savefig(f"{card}_part_e_thermal.png", dpi=150)
    print(f"Wrote {card}_part_e_thermal.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    args = ap.parse_args()
    plot_part_b(args.card)
    plot_part_d_memory(args.card)
    plot_part_e_thermal(args.card)


if __name__ == "__main__":
    main()
