"""
Generate Markdown QA reports for ``runs/<run_id>`` using manifest + metric JSONs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import EvalConfig, load_eval_config


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def dataframe_from_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        record = {"checkpoint": Path(row.get("checkpoint", \"n/a\")).name}
        record.update(metrics)
        records.append(record)
    return pd.DataFrame.from_records(records)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No metrics recorded yet._"
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines: List[str] = []
    for _, row in df.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if pd.isna(value):
                cells.append("NA")
            else:
                cells.append(f"{value}")
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, separator, *body_lines])


def plot_metrics(df: pd.DataFrame, destination: Path) -> Path | None:
    metric_columns = [col for col in df.columns if col != "checkpoint"]
    if df.empty or not metric_columns:
        return None
    melted = df.melt(id_vars="checkpoint", value_vars=metric_columns, var_name="metric", value_name="value")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=melted, x="checkpoint", y="value", hue="metric", marker="o", ax=ax)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Metric value")
    ax.set_title("QA Metric Trends")
    fig.autofmt_xdate(rotation=30)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def synthesize_metrics(config: EvalConfig) -> List[Dict[str, Any]]:
    checkpoints = config.checkpoints or [
        f"runs/{config.run_id}/checkpoints/base.pt",
        f"runs/{config.run_id}/checkpoints/permutations_only.pt",
        f"runs/{config.run_id}/checkpoints/permutations_rotations.pt",
        f"runs/{config.run_id}/checkpoints/permutations_rotations_style.pt",
    ]
    rows: List[Dict[str, Any]] = []
    metric_flags = config.metrics.to_dict()
    for idx, checkpoint in enumerate(checkpoints):
        metrics = _stub_metrics(metric_flags, idx)
        rows.append(
            {
                "run_id": config.run_id,
                "checkpoint": checkpoint,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "metric_flags": metric_flags,
                "variant_flags": config.variant.to_dict(),
                "metrics": metrics,
                "notes": config.notes or "mock metrics synthesized for report templating",
            }
        )
    return rows


def _stub_metrics(metric_flags: Mapping[str, bool], idx: int) -> Dict[str, float | None]:
    import numpy as np

    rng = np.random.default_rng(seed=idx)
    stub: Dict[str, float | None] = {}
    if metric_flags.get("logit_kl", False):
        stub["logit_kl"] = round(float(rng.uniform(0.05, 0.35)), 4)
    if metric_flags.get("hidden_state_cosine", False):
        stub["hidden_state_cosine"] = round(float(rng.uniform(0.6, 0.95)), 4)
    if metric_flags.get("residual_rms", False):
        stub["residual_rms"] = round(float(rng.uniform(0.8, 1.1)), 4)
    if metric_flags.get("surprisal", False):
        stub["surprisal"] = round(float(rng.uniform(10.0, 30.0)), 2)
    return stub


def write_metrics_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    df: pd.DataFrame,
    plot_path: Path | None,
    report_path: Path,
    manifest_path: Path,
    metrics_path: Path,
) -> None:
    created_at = manifest.get("created_at", "unknown")
    dataset_snapshot = manifest.get("dataset_snapshot_path", "n/a")
    token_info = manifest.get("tokenizer", {})
    prompt_count = len(manifest.get("prompt_hashes", []))
    notes = rows[0].get("notes", "") if rows else ""
    variant_flags = rows[0].get("variant_flags", {}) if rows else {}
    metric_flags = rows[0].get("metric_flags", {}) if rows else {}
    if plot_path:
        try:
            figure_rel = plot_path.relative_to(report_path.parent)
        except ValueError:
            figure_rel = plot_path
        figure_line = f"![Metric trends]({figure_rel})"
    else:
        figure_line = "_No plot generated yet._"

    lines = [
        f"# Run {manifest.get('run_id', 'unknown')} QA Report",
        "",
        f"- **Manifest**: `{manifest_path}` (created {created_at})",
        f"- **Dataset snapshot**: `{dataset_snapshot}`",
        f"- **Tokenizer**: {token_info}",
        f"- **Prompt files hashed**: {prompt_count}",
        f"- **Metrics JSONL**: `{metrics_path}`",
        "",
        "## Variant Toggles",
        "```json",
        json.dumps(variant_flags, indent=2),
        "```",
        "",
        "## Metric Toggles",
        "```json",
        json.dumps(metric_flags, indent=2),
        "```",
        "",
        "## Metric Table",
        dataframe_to_markdown(df),
        "",
        "## Plots",
        figure_line,
        "",
    ]
    if notes:
        lines.extend(["## Notes", notes, ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a QA report for a run directory.")
    parser.add_argument("--run-id", required=True, help="Run identifier (e.g., run_2024_....).")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing run folders.")
    parser.add_argument("--manifest", default=None, help="Override manifest path.")
    parser.add_argument("--metrics", default=None, help="Override metrics JSONL path.")
    parser.add_argument("--report", default=None, help="Override report.md output path.")
    parser.add_argument("--config", default=None, help="Optional config path to seed toggles.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Synthesize metrics if no JSONL rows exist (useful before QA runs).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.runs_root) / args.run_id
    manifest_path = Path(args.manifest) if args.manifest else run_dir / "manifest.json"
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "logs" / "metrics.jsonl"
    report_path = Path(args.report) if args.report else run_dir / "report.md"
    figures_dir = run_dir / "figures"
    plot_path = figures_dir / "metric_trends.png"

    overrides: Dict[str, Any] = {
        "run_id": args.run_id,
        "manifest_path": str(manifest_path),
        "metrics_output_path": str(metrics_path),
    }
    config = load_eval_config(args.config, overrides=overrides)

    manifest = load_manifest(manifest_path)
    rows = load_metrics(metrics_path)

    if not rows and args.mock:
        rows = synthesize_metrics(config)
        write_metrics_jsonl(rows, metrics_path)

    if not rows:
        raise RuntimeError(
            "No metrics rows were found. Run the QA notebook or pass --mock to synthesize placeholders."
        )

    df = dataframe_from_rows(rows)
    plot_file = plot_metrics(df, plot_path)
    build_report(manifest, rows, df, plot_file, report_path, manifest_path, metrics_path)
    print(f"Wrote report to {report_path}")
    if plot_file:
        print(f"- Plot saved to {plot_file}")
    print(f"- Metrics source: {metrics_path}")


if __name__ == "__main__":
    main()


