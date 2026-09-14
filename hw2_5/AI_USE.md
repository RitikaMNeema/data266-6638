# AI_USE.md — HW2.5

## Tool used
Claude (Anthropic), web/chat interface, single continuous conversation covering
setup through data collection.

## What AI assistance was used for

**Benchmark script authoring (Parts A–E).** Claude wrote the Python scripts
(`utils.py`, `part_a_provenance.py`, `part_b_precision.py`,
`part_c_bandwidth.py`, `part_d_attention.py`, `part_e_thermal.py`,
`make_plots.py`) that implement the measurement logic specified in the
assignment: dense matmul throughput sweeps across precisions, memory-bound
vs. compute-bound bandwidth/arithmetic-intensity comparisons, naive vs.
fused attention with OOM boundary search, and sustained-load thermal
sampling. I reviewed and ran every script myself on the lab machine; no
script was used unmodified without my running it and inspecting the output.

**Environment setup troubleshooting.** The lab Windows machine had no
Python, pip, or git on PATH. Claude walked me through diagnosing this
(`where python`, checking for Anaconda, `where /R` full-disk search),
installing Python and Git via `winget`, and fixing PATH persistence with
`setx`. Claude also diagnosed that the default `pip install torch` command
installs a CPU-only build on Windows and that the correct CUDA-enabled
build requires `--index-url https://download.pytorch.org/whl/cu130` to
match this driver's CUDA 13.2.

**Bug fixing.** Claude's own code verification (`py_compile`) caught a
malformed nested f-string in `part_d_attention.py` before I ever ran it,
which I would not have caught by eye. I ran the compile check myself and
confirmed the fix.

**Debugging an unexpected result.** During Part D, naive attention never
raised `OutOfMemoryError` even past 32GB of reported peak memory on a
24GB card. Claude identified this as Windows WDDM's driver-level
"shared GPU memory" fallback (spilling allocations into system RAM
instead of raising CUDA OOM), evidenced by the latency cliff at
S=28672-32768 (65ms → 325ms → 1173ms) while reported memory exceeded
physical VRAM. Claude proposed and I applied a fix
(`torch.cuda.set_per_process_memory_fraction`) to cap the allocator and
force a genuine OOM within physical VRAM bounds, which produced a clean
single-token-resolution boundary (26832 OK / 26833 OOM) via the existing
binary-search refinement logic.

**Vendor specification lookup.** Claude web-searched and cited NVIDIA's
own Ada GPU Architecture whitepaper for the RTX 4090's architecture,
memory type/bandwidth, tensor core generation, and supported reduced
precisions, required for Part A.

**Result interpretation checks (not conclusions).** At each stage Claude
flagged what a sane result should look like before I ran the next step
(e.g., BF16/FP16 should substantially exceed FP32 TFLOPS; bandwidth
should land in 80-95% of spec; fused attention should survive to much
larger sequence lengths than naive before any memory ceiling). I
confirmed these against my own actual output at each step rather than
accepting them uncritically.

## What was not AI-generated
All numerical results (TFLOPS, GB/s, memory footprints, OOM boundaries,
clock/temperature/power samples) came from running the scripts on my own
RTX 4090 GPU-lab reservation — none of these values were generated,
estimated, or fabricated by the AI. The GPU UUID, driver version, and
CUDA version in the provenance capture are direct output of `nvidia-smi`
on the lab machine, not AI-supplied. [Section on METRICS.md analysis
prose to be added once written.]
