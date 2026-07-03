from __future__ import annotations

import argparse


def run_benchmark(*args, **kwargs):
    from draftverifybench.runner import run_benchmark

    return run_benchmark(*args, **kwargs)


def _run(args: argparse.Namespace) -> None:
    run_benchmark(
        args.config,
        out=args.out,
        raw_out=args.raw_out,
        metadata_out=args.metadata_out,
        max_prompts=args.max_prompts,
    )


def _summarize(args: argparse.Namespace) -> None:
    from draftverifybench.analysis import summarize_results

    print(summarize_results(args.results).to_string(index=False))


def _plot(args: argparse.Namespace) -> None:
    import sys

    from scripts.make_plots import main as plot_main

    sys.argv = ["make_plots", "--inputs", args.results, "--out-dir", args.out_dir]
    plot_main()


def _router(args: argparse.Namespace) -> None:
    from scripts.run_router_experiment import run_router_experiment

    run_router_experiment(
        args.config,
        out=args.out,
        raw_out=args.raw_out,
        metadata_out=args.metadata_out,
        max_prompts=args.max_prompts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="draftverifybench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a benchmark config.")
    run.add_argument("--config", required=True)
    run.add_argument("--out", default="results/benchmark_results.csv")
    run.add_argument("--raw-out", default="results/benchmark_raw.jsonl")
    run.add_argument("--metadata-out", default="results/benchmark_metadata.json")
    run.add_argument("--max-prompts", type=int)
    run.set_defaults(func=_run)

    adaptive = sub.add_parser("adaptive", help="Run an adaptive benchmark config.")
    adaptive.add_argument("--config", default="configs/adaptive_local.yaml")
    adaptive.add_argument("--out", default="results/adaptive_local_results.csv")
    adaptive.add_argument("--raw-out", default="results/adaptive_local_raw.jsonl")
    adaptive.add_argument("--metadata-out", default="results/adaptive_local_metadata.json")
    adaptive.add_argument("--max-prompts", type=int)
    adaptive.set_defaults(func=_run)

    summarize = sub.add_parser("summarize", help="Summarize result CSV.")
    summarize.add_argument("--results", required=True)
    summarize.set_defaults(func=_summarize)

    plot = sub.add_parser("plot", help="Create plots from result CSV.")
    plot.add_argument("--results", required=True)
    plot.add_argument("--out-dir", default="results/plots")
    plot.set_defaults(func=_plot)

    router = sub.add_parser("router", help="Run prompt-router benchmark.")
    router.add_argument("--config", default="configs/router_local.yaml")
    router.add_argument("--out", default="results/router_local_results.csv")
    router.add_argument("--raw-out", default="results/router_local_raw.jsonl")
    router.add_argument("--metadata-out", default="results/router_local_metadata.json")
    router.add_argument("--max-prompts", type=int)
    router.set_defaults(func=_router)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
