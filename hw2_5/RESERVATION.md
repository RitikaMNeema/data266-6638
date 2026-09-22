# RTX 4090 Workstation — Reservation Record

## Reservation details

| Field | Value |
|---|---|
| Room | ISB 840 |
| Machine No. | 25 |
| Reservation type | Team reservation (approval issued to Rajesh Paruchuri) |
| Reservation start | Monday, September 14, 2026, 12:00 PM PDT |
| Reservation end | Tuesday, September 15, 2026, 11:59 AM PDT |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU UUID | GPU-5b052ad1-4272-40db-4b25-c930bf32b547 |
| Driver / CUDA | 595.95 / 13.2 |

## My time at the workstation (Monday, Sep 14, 2026)

| Event | Time (PDT) |
|---|---|
| Entered lab | 2:00 PM |
| Started working | 2:10 PM |
| `nvidia-smi -q` provenance capture (Part A, GPU idle: 34°C, P8, 5% util) | 2:48:12 PM |
| Logged out / left | 5:10 PM |

**Time at the workstation:** ~3 h 10 min (2:00 PM – 5:10 PM), of which ~3 h 00 min was active work (2:10 PM – 5:10 PM).

## Notes

- The reservation itself was booked as a team slot under Rajesh Paruchuri's name for Room ISB 840 / Machine 25; my individual session on that machine ran 2:00–5:10 PM on the first day of the 24-hour window.
- The provenance snapshot (`rtx4090_nvidia_smi_q.txt`) was captured at 2:48:12 PM, ~38 minutes after I started working — consistent with environment setup (Python/CUDA checks) preceding the Part A capture.
- GPU-busy time (as opposed to time physically at the workstation) is not separately logged here beyond the idle provenance snapshot above; the benchmark timings themselves (Parts B–E) are reported against wall-clock durations in `METRICS.md` / `hw25_report.md`, not against a separate login/logout GPU-hours meter.
