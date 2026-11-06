from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .benchmark import benchmark, save_report
from .picmus import benchmark_picmus
from .workload import load_pymust_doppler_workload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beamform-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("benchmark", help="beamform a raw-RF ensemble")
    run.add_argument("--dataset", choices=("pymust", "picmus"), default="pymust")
    run.add_argument("--mode", choices=("numpy", "cupy"), default="numpy")
    run.add_argument("--runs", type=int)
    run.add_argument("--warmup", type=int)
    run.add_argument("--batch-size", type=int, help="PyMUST frame batch size")
    run.add_argument("--grid-size", type=int, help="PyMUST scan grid width/depth")
    run.add_argument("--deadline-ms", type=float)
    run.add_argument(
        "--measure-cold-start",
        action="store_true",
        help=(
            "Measure device setup and the first PyMUST ensemble before warmups; "
            "report separately from steady-state latency"
        ),
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--save-power", type=Path)
    run.add_argument("--save-image", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.deadline_ms is not None and args.deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")

    if args.dataset == "pymust":
        if args.save_image is not None:
            raise ValueError("--save-image is only available with --dataset picmus")
        workload = load_pymust_doppler_workload(
            grid_size=args.grid_size if args.grid_size is not None else 251
        )
        report, image = benchmark(
            workload,
            mode=args.mode,
            runs=args.runs if args.runs is not None else 200,
            warmup=args.warmup if args.warmup is not None else 10,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            deadline_ms=args.deadline_ms,
            measure_cold_start=args.measure_cold_start,
        )
    else:
        if args.measure_cold_start:
            raise ValueError(
                "--measure-cold-start is currently supported only with "
                "--dataset pymust"
            )
        if args.batch_size is not None or args.grid_size is not None:
            raise ValueError(
                "--batch-size and --grid-size apply only to --dataset pymust; "
                "PICMUS uses the native scan and compounds all transmit angles."
            )
        if args.save_power is not None:
            raise ValueError("--save-power is only available with --dataset pymust")
        report, image = benchmark_picmus(
            mode=args.mode,
            runs=args.runs if args.runs is not None else 20,
            warmup=args.warmup if args.warmup is not None else 2,
            deadline_ms=args.deadline_ms,
        )

    print(json.dumps(report, indent=2))
    if args.output is not None:
        save_report(report, args.output)
        print(f"Report saved: {args.output}")
    save_path = args.save_power if args.save_power is not None else args.save_image
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, image)
        output_name = "Power Doppler" if args.save_power is not None else "B-mode magnitude"
        print(f"{output_name} array saved: {save_path}")


if __name__ == "__main__":
    main()
