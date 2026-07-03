from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftverifybench.analysis import (
    build_summary_tables,
    correlation_analysis,
    load_results,
    output_equivalence,
)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.copy()
    if len(display) > 30:
        display = display.head(30)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def write_summary(inputs: list[str], out: str, tables_out: str) -> None:
    df = load_results(inputs)
    tables = build_summary_tables(df)
    tables_dir = Path(tables_out)
    tables_dir.mkdir(parents=True, exist_ok=True)

    md = [
        "# DraftVerifyBench Summary",
        "",
        f"Inputs: {', '.join(inputs)}",
        f"Rows after error filtering: {len(df)}",
        "",
    ]
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
        md.extend([f"## {name}", "", _markdown_table(table), ""])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(md), encoding="utf-8")


def write_correlation(inputs: list[str], out_md: str, out_csv: str) -> None:
    df = load_results(inputs)
    corr = correlation_analysis(df)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, index=False)
    md = [
        "# Correlation Analysis",
        "",
        "Correlation values are Pearson correlations over successful speculative rows.",
        "",
        _markdown_table(corr),
    ]
    Path(out_md).write_text("\n".join(md), encoding="utf-8")


def write_equivalence(raw_jsonl: str, results_csv: str, out_md: str) -> None:
    summary = output_equivalence(raw_jsonl, results_csv)
    lines = [
        "# Output Equivalence Check",
        "",
        "This check applies to greedy decoding only (`temperature == 0.0`).",
        "",
        f"Greedy speculative rows: {summary['greedy_speculative_rows']}",
        f"Exact output match rate: {summary['exact_match_rate']:.4f}",
        f"Token-level match rate: {summary['token_level_match_rate']:.4f}",
        "",
        "Speculative decoding should match the verifier's greedy output when accepted/rejected "
        "tokens are handled correctly. Non-zero-temperature rows are excluded because exact "
        "speculative sampling with distribution correction is not implemented.",
        "",
        "## Differences",
        "",
    ]
    diffs = summary["differences"]
    if diffs:
        for diff in diffs:
            lines.append(f"- {diff}")
    else:
        lines.append("No greedy output differences were observed.")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DraftVerifyBench result CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tables-out", required=True)
    parser.add_argument("--correlation-out")
    parser.add_argument("--correlation-csv")
    parser.add_argument("--raw-jsonl")
    parser.add_argument("--equivalence-out")
    args = parser.parse_args()
    write_summary(args.inputs, args.out, args.tables_out)
    if args.correlation_out and args.correlation_csv:
        write_correlation(args.inputs, args.correlation_out, args.correlation_csv)
    if args.raw_jsonl and args.equivalence_out:
        write_equivalence(args.raw_jsonl, args.inputs[0], args.equivalence_out)


if __name__ == "__main__":
    main()
