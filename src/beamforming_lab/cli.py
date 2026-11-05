from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .benchmark import benchmark, save_report
from .workload import load_pymust_doppler_workload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beamform-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("benchmark", help="beamform a raw-RF ensemble")
    run.add_argument("--mode", choices=("numpy", "cupy"), default="numpy")
    run.add_argument("--runs", type=int, default=200)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--grid-size", type=int, default=251)
    run.add_argument("--deadline-ms", type=float)
    run.add_argument("--output", type=Path)
    run.add_argument("--save-power", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.deadline_ms is not None and args.deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")

    workload = load_pymust_doppler_workload(grid_size=args.grid_size)
    report, power_doppler = benchmark(
        workload,
        mode=args.mode,
        runs=args.runs,
        warmup=args.warmup,
        batch_size=args.batch_size,
        deadline_ms=args.deadline_ms,
    )
    print(json.dumps(report, indent=2))
    if args.output is not None:
        save_report(report, args.output)
        print(f"Report saved: {args.output}")
    if args.save_power is not None:
        args.save_power.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_power, power_doppler)
        print(f"Power Doppler array saved: {args.save_power}")


if __name__ == "__main__":
    main()
