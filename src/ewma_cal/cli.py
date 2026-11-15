"""Prepare private inputs, replay the frozen strategy, or inspect saved results."""

import argparse
from pathlib import Path

from .config import ModelConfig
from .data import MarketBundle
from .reporting import build_report, comparison_table, load_reference


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ewma-cal")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference", help="Display committed aggregate results")
    reference.add_argument("--root", type=Path, default=Path.cwd())
    reference.add_argument("--period", choices=["Full", "Development", "OOS"], default="Full")
    prep = commands.add_parser("prepare", help="Build a private bundle from a licensed archive")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--end", default="2026-11-10")
    benchmark = commands.add_parser(
        "benchmark", help="Download SPY adjusted closes (optional network operation)"
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--start", default="2015-12-31")
    benchmark.add_argument("--end", default="2026-11-11", help="Exclusive end date")
    report = commands.add_parser("report", help="Replay private inputs with frozen parameters")
    report.add_argument("--inputs", type=Path, required=True)
    report.add_argument("--benchmark", type=Path, required=True)
    report.add_argument("--config", type=Path, default=Path("config/model.json"))
    report.add_argument("--output", type=Path, required=True)
    for name in ["sized-report", "leverage-test"]:
        command = commands.add_parser(
            name, help="Replay the frozen funded sizing or test exposure targets"
        )
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--benchmark", type=Path, required=True)
        command.add_argument("--config", type=Path, default=Path("config/model.json"))
        command.add_argument("--sizing", type=Path, default=Path("config/sizing.json"))
        command.add_argument("--output", type=Path, required=True)
        if name == "leverage-test":
            command.add_argument("--scales", default="0.5,1,2,3,4,6,8,10,12,16")
    args = parser.parse_args(argv)
    if args.command == "reference":
        print(
            comparison_table(load_reference(args.root), args.period).to_string(float_format="%.4f")
        )
    elif args.command == "prepare":
        from .preparation import prepare

        prepare(args.data_root, args.output, args.end)
    elif args.command == "benchmark":
        from .benchmark import download

        download(args.output, args.start, args.end)
    elif args.command == "report":
        result = build_report(
            MarketBundle.load(args.inputs),
            ModelConfig.load(args.config),
            args.benchmark,
            args.output,
        )
        print(result[result.Primary].to_string(index=False))
    else:
        from .sizing import SizingConfig

        bundle = MarketBundle.load(args.inputs)
        config = ModelConfig.load(args.config)
        sizing = SizingConfig.load(args.sizing)
        if args.command == "sized-report":
            result = build_report(bundle, config, args.benchmark, args.output, sizing=sizing)
            print(result[result.Primary].to_string(index=False))
        else:
            from .leverage import check_freeze, run_leverage_test

            root = args.sizing.resolve().parent.parent
            check_freeze(root)
            result = run_leverage_test(
                bundle,
                config,
                sizing,
                args.benchmark,
                args.output,
                scales=[float(x) for x in args.scales.split(",")],
            )
            check_freeze(root)
            print(
                result[(result.Period == "Full") & ~result.CashInterest][
                    ["TargetScale", "CAGR", "Sharpe", "MaxDrawdown", "Trades"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
