"""
Part A — Onboarding and provenance.

Run once per machine (RTX 5090 box, then RTX 4090 box). Captures the full
nvidia-smi -q text (commit this file as-is) and writes a small JSON summary
of the fields the assignment asks you to record by hand: UUID, driver
version, CUDA version, VRAM capacity, power limit.

The vendor spec fields (architecture, memory type/bandwidth, tensor core
generation, supported reduced precisions) are NOT auto-fillable — go look
them up in the NVIDIA Ada/Blackwell whitepapers or ark.intel-style product
pages and fill provenance_manual.json yourself, with a source URL per card.
That's a citation requirement, not a scraping problem.

Usage: python part_a_provenance.py --out-prefix rtx5090
"""
import argparse
import json
import re
import subprocess


def capture_nvidia_smi_q(out_path: str) -> str:
    text = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    with open(out_path, "w") as f:
        f.write(text)
    return text


def parse_summary(smi_text: str) -> dict:
    def grab(pattern, cast=str):
        m = re.search(pattern, smi_text)
        return cast(m.group(1).strip()) if m else None

    return {
        "gpu_uuid": grab(r"GPU UUID\s*:\s*(GPU-[0-9a-fA-F-]+)"),
        "driver_version": grab(r"Driver Version\s*:\s*([\d.]+)"),
        "cuda_version": grab(r"CUDA Version\s*:\s*([\d.]+)"),
        "vram_total": grab(r"FB Memory Usage\s*\n\s*Total\s*:\s*(.+)"),
        "power_limit": grab(r"Power Limit\s*:\s*(.+)"),
        "product_name": grab(r"Product Name\s*:\s*(.+)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", required=True,
                     help="e.g. rtx5090 -> writes rtx5090_nvidia_smi_q.txt "
                          "and rtx5090_provenance_auto.json")
    args = ap.parse_args()

    smi_path = f"{args.out_prefix}_nvidia_smi_q.txt"
    smi_text = capture_nvidia_smi_q(smi_path)
    summary = parse_summary(smi_text)

    json_path = f"{args.out_prefix}_provenance_auto.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {smi_path} (commit this raw file)")
    print(f"Wrote {json_path}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    missing = [k for k, v in summary.items() if v is None]
    if missing:
        print(f"WARNING: could not parse {missing} — check your driver's "
              f"nvidia-smi -q output format and adjust the regex.")


if __name__ == "__main__":
    main()
