import argparse
import csv
import subprocess
import sys
import time
from typing import Dict, List, Optional

import torch


def parse_sizes_mb(text: str) -> List[int]:
    out = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        v = int(p)
        if v <= 0:
            raise ValueError(f"invalid size: {v}")
        out.append(v)
    if not out:
        raise ValueError("sizes_mb is empty")
    return out


def gpu_info_from_torch(device_id: int) -> Dict[str, str]:
    props = torch.cuda.get_device_properties(device_id)
    return {
        "name": str(props.name),
        "sm": f"{props.major}.{props.minor}",
        "total_mem_gb": f"{props.total_memory / (1 << 30):.2f}",
    }


def gpu_info_from_nvidia_smi(device_id: int) -> Dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return {}

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx != device_id:
            continue
        return {
            "pcie_gen_current": parts[2],
            "pcie_gen_max": parts[3],
            "pcie_width_current": parts[4],
            "pcie_width_max": parts[5],
        }
    return {}


def bench_one_direction(
    direction: str,
    size_bytes: int,
    iters: int,
    warmup: int,
    device: torch.device,
    pinned: bool,
) -> Dict[str, float]:
    assert direction in ("h2d", "d2h", "d2d")
    n = size_bytes

    if direction == "d2d":
        src = torch.empty(n, dtype=torch.uint8, device=device)
        dst = torch.empty_like(src)
    else:
        h = torch.empty(n, dtype=torch.uint8, device="cpu", pin_memory=pinned)
        d = torch.empty(n, dtype=torch.uint8, device=device)

    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if direction == "h2d":
                d.copy_(h, non_blocking=True)
            elif direction == "d2h":
                h.copy_(d, non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            if direction == "h2d":
                d.copy_(h, non_blocking=True)
            elif direction == "d2h":
                h.copy_(d, non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)
        end.record(stream)
    stream.synchronize()
    ms = float(start.elapsed_time(end))
    gb = (size_bytes * iters) / 1e9
    gbps = gb / (ms / 1e3)
    return {
        "size_mb": size_bytes / (1 << 20),
        "total_ms": ms,
        "gbps": gbps,
    }


def bench_duplex(
    size_bytes: int,
    iters: int,
    warmup: int,
    device: torch.device,
    pinned: bool,
) -> Dict[str, float]:
    h1 = torch.empty(size_bytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)
    h2 = torch.empty(size_bytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)
    d1 = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    d2 = torch.empty(size_bytes, dtype=torch.uint8, device=device)

    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)

    for _ in range(warmup):
        with torch.cuda.stream(s1):
            d1.copy_(h1, non_blocking=True)
        with torch.cuda.stream(s2):
            h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.cuda.stream(s1):
            d1.copy_(h1, non_blocking=True)
        with torch.cuda.stream(s2):
            h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    sec = max(t1 - t0, 1e-12)
    gb = (size_bytes * iters * 2) / 1e9
    gbps = gb / sec
    return {
        "size_mb": size_bytes / (1 << 20),
        "total_ms": sec * 1e3,
        "gbps": gbps,
    }


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 2

    torch.cuda.set_device(args.device)
    dev = torch.device(f"cuda:{args.device}")
    sizes_mb = parse_sizes_mb(args.sizes_mb)
    sizes_bytes = [v * (1 << 20) for v in sizes_mb]

    print("== PCIe Benchmark ==")
    print(f"device: cuda:{args.device}")
    ti = gpu_info_from_torch(args.device)
    print(f"gpu: {ti.get('name', 'unknown')}, sm={ti.get('sm', '?')}, mem={ti.get('total_mem_gb', '?')} GB")
    si = gpu_info_from_nvidia_smi(args.device)
    if si:
        print(
            "pcie: "
            f"gen {si.get('pcie_gen_current', '?')}/{si.get('pcie_gen_max', '?')}, "
            f"width x{si.get('pcie_width_current', '?')}/x{si.get('pcie_width_max', '?')}"
        )
    else:
        print("pcie: nvidia-smi query unavailable")
    print(f"sizes_mb: {sizes_mb}")
    print(f"iters={args.iters}, warmup={args.warmup}")
    print("")

    rows: List[Dict[str, object]] = []

    for size_b in sizes_bytes:
        if not args.skip_pinned:
            h2d = bench_one_direction("h2d", size_b, args.iters, args.warmup, dev, pinned=True)
            d2h = bench_one_direction("d2h", size_b, args.iters, args.warmup, dev, pinned=True)
            d2d = bench_one_direction("d2d", size_b, args.iters, args.warmup, dev, pinned=True)
            duplex = bench_duplex(size_b, args.iters, args.warmup, dev, pinned=True)
            print(
                f"[pinned] {int(h2d['size_mb']):>4} MB | "
                f"H2D {h2d['gbps']:>7.2f} GB/s | "
                f"D2H {d2h['gbps']:>7.2f} GB/s | "
                f"D2D {d2d['gbps']:>7.2f} GB/s | "
                f"duplex {duplex['gbps']:>7.2f} GB/s"
            )
            rows.extend(
                [
                    {"mem": "pinned", "dir": "h2d", **h2d},
                    {"mem": "pinned", "dir": "d2h", **d2h},
                    {"mem": "pinned", "dir": "d2d", **d2d},
                    {"mem": "pinned", "dir": "duplex_h2d_d2h", **duplex},
                ]
            )

        if not args.skip_pageable:
            h2d = bench_one_direction("h2d", size_b, args.iters, args.warmup, dev, pinned=False)
            d2h = bench_one_direction("d2h", size_b, args.iters, args.warmup, dev, pinned=False)
            print(
                f"[pagebl] {int(h2d['size_mb']):>4} MB | "
                f"H2D {h2d['gbps']:>7.2f} GB/s | "
                f"D2H {d2h['gbps']:>7.2f} GB/s"
            )
            rows.extend(
                [
                    {"mem": "pageable", "dir": "h2d", **h2d},
                    {"mem": "pageable", "dir": "d2h", **d2h},
                ]
            )

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mem", "dir", "size_mb", "total_ms", "gbps"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved CSV: {args.csv_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure PCIe copy bandwidth (A6000/A100).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--sizes_mb",
        type=str,
        default="16,64,256,1024",
        help="Comma-separated payload sizes in MB",
    )
    p.add_argument("--iters", type=int, default=100, help="Iterations per size")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations per size")
    p.add_argument("--skip_pinned", action="store_true", help="Skip pinned-memory tests")
    p.add_argument("--skip_pageable", action="store_true", help="Skip pageable-memory tests")
    p.add_argument("--csv_out", type=str, default="", help="Optional CSV output path")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    rc = run(args)
    sys.exit(rc)
