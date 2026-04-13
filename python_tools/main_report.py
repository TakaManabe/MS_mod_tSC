#!/usr/bin/env python3
"""
Build a comprehensive statistical report + figures from
directional_event_table_stim_control.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


METRIC_PAIRS = [
    ("hc_pre_rms", "hc_post_rms", "HC RMS"),
    ("hc_pre_theta_pow", "hc_post_theta_pow", "HC theta power"),
    ("ms_pre_rate_total", "ms_post_rate_total", "MS total firing rate"),
    ("ms_pre_rate_per_unit", "ms_post_rate_per_unit", "MS per-unit firing rate"),
]

DELTA_METRICS = [
    ("hc_delta_rms", "HC delta RMS"),
    ("hc_delta_theta_pow", "HC delta theta power"),
    ("ms_delta_rate_total", "MS delta total rate"),
    ("ms_delta_rate_per_unit", "MS delta per-unit rate"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="directional_event_table_stim_control.csv")
    p.add_argument("--outdir", required=True, help="output directory (analysis_directional)")
    return p.parse_args()


def clean_series(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return y.replace([np.inf, -np.inf], np.nan).dropna()


def cohens_d_paired(pre: np.ndarray, post: np.ndarray) -> float:
    diff = post - pre
    sd = np.std(diff, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n_x = len(x)
    n_y = len(y)
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    gt = np.sum(x > y)
    lt = np.sum(x < y)
    return float((gt - lt) / (n_x * n_y))


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NaN"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["font.size"] = 10


def fig_count_by_dataset_condition(df: pd.DataFrame, out: Path) -> None:
    g = (
        df.groupby(["dataset", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=g, x="dataset", y="n_events", hue="condition", ax=ax)
    ax.set_title("Event count by dataset and condition")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("N events")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_events_per_animal(df: pd.DataFrame, out: Path) -> None:
    g = (
        df.groupby(["dataset", "animal"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=g, x="animal", y="n_events", hue="dataset", ax=ax)
    ax.set_title("Events per animal")
    ax.set_xlabel("Animal")
    ax.set_ylabel("N events")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_violin_delta(df: pd.DataFrame, metric: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=df, x="dataset", y=metric, inner="quartile", cut=0, ax=ax)
    sns.stripplot(
        data=df.sample(min(4000, len(df)), random_state=7),
        x="dataset",
        y=metric,
        color="black",
        size=1.8,
        alpha=0.25,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_paired_mean_bars(df: pd.DataFrame, pre: str, post: str, title: str, out: Path) -> None:
    g = (
        df.groupby("dataset")[[pre, post]]
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars="dataset", var_name="phase", value_name="mean_value")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=g, x="dataset", y="mean_value", hue="phase", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "hc_pre_rms",
        "hc_post_rms",
        "hc_delta_rms",
        "hc_pre_theta_pow",
        "hc_post_theta_pow",
        "hc_delta_theta_pow",
        "ms_pre_rate_total",
        "ms_post_rate_total",
        "ms_delta_rate_total",
        "n_tt_units",
        "n_ms_tetrodes",
    ]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Spearman correlation heatmap")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_delta_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    samp = df.sample(min(5000, len(df)), random_state=7)
    sns.scatterplot(
        data=samp,
        x="ms_delta_rate_total",
        y="hc_delta_rms",
        hue="dataset",
        alpha=0.45,
        s=20,
        ax=ax,
    )
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    ax.set_title("MS delta rate vs HC delta RMS")
    ax.set_xlabel("MS delta total rate")
    ax.set_ylabel("HC delta RMS")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_effect_size_forest(effect_df: pd.DataFrame, out: Path) -> None:
    plot_df = effect_df.dropna(subset=["cohens_d"]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["dataset"] + " | " + plot_df["metric"]
    plot_df = plot_df.sort_values("cohens_d")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(plot_df))))
    ax.hlines(plot_df["label"], 0, plot_df["cohens_d"], color="steelblue", lw=2)
    ax.plot(plot_df["cohens_d"], plot_df["label"], "o", color="black")
    ax.axvline(0, color="gray", lw=1)
    ax.set_title("Paired effect size (Cohen's d; post - pre)")
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Dataset | Metric")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, sub in df.groupby("dataset"):
        for pre, post, label in METRIC_PAIRS:
            x = pd.to_numeric(sub[pre], errors="coerce")
            y = pd.to_numeric(sub[post], errors="coerce")
            m = x.notna() & y.notna()
            x = x[m].to_numpy()
            y = y[m].to_numpy()
            n = len(x)
            if n < 3:
                rows.append(
                    {
                        "dataset": ds,
                        "metric": label,
                        "n": n,
                        "mean_pre": np.nan,
                        "mean_post": np.nan,
                        "mean_delta": np.nan,
                        "ttest_p": np.nan,
                        "wilcoxon_p": np.nan,
                        "cohens_d": np.nan,
                    }
                )
                continue
            t_p = stats.ttest_rel(y, x, nan_policy="omit").pvalue
            try:
                w_p = stats.wilcoxon(y, x).pvalue
            except ValueError:
                w_p = np.nan
            rows.append(
                {
                    "dataset": ds,
                    "metric": label,
                    "n": n,
                    "mean_pre": float(np.mean(x)),
                    "mean_post": float(np.mean(y)),
                    "mean_delta": float(np.mean(y - x)),
                    "ttest_p": float(t_p),
                    "wilcoxon_p": float(w_p),
                    "cohens_d": cohens_d_paired(x, y),
                }
            )
    out = pd.DataFrame(rows)
    for pcol in ["ttest_p", "wilcoxon_p"]:
        mask = out[pcol].notna()
        if mask.any():
            out.loc[mask, pcol + "_fdr_bh"] = multipletests(
                out.loc[mask, pcol].to_numpy(), method="fdr_bh"
            )[1]
        else:
            out[pcol + "_fdr_bh"] = np.nan
    return out


def between_dataset_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    for col, label in DELTA_METRICS:
        groups = []
        for ds in datasets:
            s = clean_series(df.loc[df["dataset"] == ds, col])
            if len(s) > 0:
                groups.append((ds, s.to_numpy()))

        if len(groups) < 2:
            rows.append(
                {
                    "metric": label,
                    "kw_p": np.nan,
                    "pair": "",
                    "mw_p": np.nan,
                    "cliffs_delta": np.nan,
                }
            )
            continue

        kw_p = stats.kruskal(*[g[1] for g in groups]).pvalue
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a_name, a = groups[i]
                b_name, b = groups[j]
                try:
                    mw_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                except ValueError:
                    mw_p = np.nan
                rows.append(
                    {
                        "metric": label,
                        "kw_p": float(kw_p),
                        "pair": f"{a_name} vs {b_name}",
                        "mw_p": float(mw_p) if np.isfinite(mw_p) else np.nan,
                        "cliffs_delta": cliffs_delta(a, b),
                    }
                )
    out = pd.DataFrame(rows)
    mask = out["mw_p"].notna()
    if mask.any():
        out.loc[mask, "mw_p_fdr_bh"] = multipletests(
            out.loc[mask, "mw_p"].to_numpy(), method="fdr_bh"
        )[1]
    else:
        out["mw_p_fdr_bh"] = np.nan
    return out


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tbl = {}
    tbl["dataset_condition_counts"] = (
        df.groupby(["dataset", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    tbl["subject_counts"] = (
        df.groupby(["dataset", "animal"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    tbl["session_counts"] = (
        df.groupby(["dataset", "animal", "session"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    tbl["dataset_metric_summary"] = (
        df.groupby("dataset")[
            [
                "hc_delta_rms",
                "hc_delta_theta_pow",
                "ms_delta_rate_total",
                "ms_delta_rate_per_unit",
            ]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    # flatten multi-index columns
    flat = []
    for c in tbl["dataset_metric_summary"].columns:
        if isinstance(c, tuple):
            flat.append("_".join([x for x in c if x]))
        else:
            flat.append(c)
    tbl["dataset_metric_summary"].columns = flat
    return tbl


def write_markdown_summary(
    out_md: Path,
    df: pd.DataFrame,
    paired_df: pd.DataFrame,
    between_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    n_rows = len(df)
    n_datasets = df["dataset"].nunique(dropna=True)
    n_animals = df[["dataset", "animal"]].drop_duplicates().shape[0]
    n_sessions = df[["dataset", "animal", "session"]].drop_duplicates().shape[0]
    cond_counts = tables["dataset_condition_counts"]

    lines = []
    lines.append("# Directional Analysis Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total event rows: **{n_rows}**")
    lines.append(f"- Datasets: **{n_datasets}**")
    lines.append(f"- Dataset-animal combinations: **{n_animals}**")
    lines.append(f"- Sessions: **{n_sessions}**")
    lines.append("")
    lines.append("### Events by dataset and condition")
    for _, r in cond_counts.iterrows():
        lines.append(
            f"- `{r['dataset']}` | `{r['condition']}`: {int(r['n_events'])} events"
        )
    lines.append("")
    if (df["condition"] == "stim").sum() == 0:
        lines.append(
            "> Note: No `stim` events were found in this table. Current analysis is control-only."
        )
        lines.append("")

    lines.append("## Paired pre-vs-post tests (within event)")
    for _, r in paired_df.iterrows():
        lines.append(
            f"- `{r['dataset']}` | {r['metric']} | n={int(r['n'])} | "
            f"delta={r['mean_delta']:.4g} | "
            f"ttest p={fmt_p(r['ttest_p'])}, q={fmt_p(r.get('ttest_p_fdr_bh', np.nan))} | "
            f"wilcoxon p={fmt_p(r['wilcoxon_p'])}, q={fmt_p(r.get('wilcoxon_p_fdr_bh', np.nan))} | "
            f"d={r['cohens_d']:.3g}"
        )
    lines.append("")

    lines.append("## Between-dataset tests (delta metrics)")
    for metric, sub in between_df.groupby("metric", dropna=False):
        kw = sub["kw_p"].dropna()
        kw_p = kw.iloc[0] if not kw.empty else np.nan
        lines.append(f"- {metric}: Kruskal-Wallis p={fmt_p(kw_p)}")
        for _, r in sub.iterrows():
            if not r.get("pair"):
                continue
            lines.append(
                f"  - {r['pair']}: MW p={fmt_p(r['mw_p'])}, q={fmt_p(r.get('mw_p_fdr_bh', np.nan))}, "
                f"Cliff's delta={r['cliffs_delta']:.3g}"
            )
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_style()
    df = pd.read_csv(in_csv)

    # numeric coercion for critical columns
    numeric_cols = [
        "event_sample",
        "hc_channel",
        "n_tt_units",
        "n_ms_tetrodes",
        "hc_pre_rms",
        "hc_post_rms",
        "hc_delta_rms",
        "hc_pre_theta_pow",
        "hc_post_theta_pow",
        "hc_delta_theta_pow",
        "ms_pre_spike_count",
        "ms_post_spike_count",
        "ms_delta_spike_count",
        "ms_pre_rate_total",
        "ms_post_rate_total",
        "ms_delta_rate_total",
        "ms_pre_rate_per_unit",
        "ms_post_rate_per_unit",
        "ms_delta_rate_per_unit",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # tables
    tables = build_summary_tables(df)
    for name, tdf in tables.items():
        save_table(tdf, outdir / f"{name}.csv")

    paired_df = paired_tests(df)
    between_df = between_dataset_tests(df)
    save_table(paired_df, outdir / "paired_prepost_tests.csv")
    save_table(between_df, outdir / "between_dataset_tests.csv")

    # figures
    fig_count_by_dataset_condition(df, outdir / "fig01_dataset_condition_counts.png")
    fig_events_per_animal(df, outdir / "fig02_events_per_animal.png")
    fig_violin_delta(df, "hc_delta_rms", "HC delta RMS by dataset", outdir / "fig03_hc_delta_rms_violin.png")
    fig_violin_delta(df, "hc_delta_theta_pow", "HC delta theta power by dataset", outdir / "fig04_hc_delta_theta_violin.png")
    fig_violin_delta(df, "ms_delta_rate_total", "MS delta total rate by dataset", outdir / "fig05_ms_delta_rate_violin.png")
    fig_paired_mean_bars(df, "hc_pre_rms", "hc_post_rms", "HC RMS: pre vs post mean", outdir / "fig06_hc_prepost_mean.png")
    fig_paired_mean_bars(df, "ms_pre_rate_total", "ms_post_rate_total", "MS total rate: pre vs post mean", outdir / "fig07_ms_prepost_mean.png")
    fig_correlation_heatmap(df, outdir / "fig08_correlation_heatmap.png")
    fig_delta_scatter(df, outdir / "fig09_ms_vs_hc_delta_scatter.png")
    fig_effect_size_forest(paired_df, outdir / "fig10_effect_size_forest.png")

    write_markdown_summary(
        outdir / "directional_report_summary.md",
        df,
        paired_df,
        between_df,
        tables,
    )

    print(f"[OK] report written to: {outdir}")


if __name__ == "__main__":
    main()
