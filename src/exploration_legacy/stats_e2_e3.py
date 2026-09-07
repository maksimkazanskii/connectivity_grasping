#!/usr/bin/env python3
"""
Compare E2 (known-success structural reconstruction) and E3
(hidden-outcome active querying) from their checkpoint CSVs.

Designed to be safe while runs are still incomplete:
- reads whatever completed-object rows currently exist;
- never assumes 200 objects are finished;
- reports E2/E3 separately;
- compares only object names present in BOTH experiments;
- paired tests use only common objects for which both methods reached the target;
- explicitly labels the analysis PARTIAL when object counts are below --expected-objects.

Default result locations match the two supplied scripts:
    E2: results/online_connectivity/online_connectivity_matches.csv
    E3: results/active_connectivity_e3/online_connectivity_matches.csv

Example:
    python compare_e2_e3_stats.py

Or:
    python compare_e2_e3_stats.py \
        --e2 results/online_connectivity/online_connectivity_matches.csv \
        --e3 results/active_connectivity_e3/online_connectivity_matches.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REQUIRED_COLUMNS = {
    "object",
    "random_target",
    "random_error",
    "connectivity_points",
    "connectivity_saving",
    "connectivity_speedup",
    "fps_points",
    "fps_saving",
}


def load_matches(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} results not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} file is missing columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep the last row if an interrupted/restarted run created duplicates.
    df = df.drop_duplicates(subset=["object", "random_target"], keep="last").copy()
    df["random_target"] = pd.to_numeric(df["random_target"], errors="coerce")

    # Recompute these from the primitive columns when possible, so the comparison
    # does not depend on stale derived columns in an interrupted checkpoint.
    reached = df["connectivity_points"].notna()
    df.loc[reached, "connectivity_speedup"] = (
            df.loc[reached, "random_target"]
            / df.loc[reached, "connectivity_points"]
    )
    df.loc[reached, "connectivity_saving"] = (
            1.0
            - df.loc[reached, "connectivity_points"]
            / df.loc[reached, "random_target"]
    )

    return df


def q25(x):
    return np.nanpercentile(x, 25) if len(x) else np.nan


def q75(x):
    return np.nanpercentile(x, 75) if len(x) else np.nan


def experiment_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []

    for target in sorted(df["random_target"].dropna().unique()):
        x = df[df["random_target"] == target].copy()
        conn = x[x["connectivity_points"].notna()].copy()
        fps = x[x["fps_points"].notna()].copy()

        n = x["object"].nunique()
        n_conn = conn["object"].nunique()
        n_fps = fps["object"].nunique()

        rows.append({
            "experiment": label,
            "target": int(target),
            "objects_completed": n,
            "conn_reached_n": n_conn,
            "conn_reach_%": 100.0 * n_conn / n if n else np.nan,
            "conn_points_mean": conn["connectivity_points"].mean(),
            "conn_points_median": conn["connectivity_points"].median(),
            "conn_points_q25": q25(conn["connectivity_points"].dropna().to_numpy()),
            "conn_points_q75": q75(conn["connectivity_points"].dropna().to_numpy()),
            "speedup_mean_x": conn["connectivity_speedup"].mean(),
            "speedup_median_x": conn["connectivity_speedup"].median(),
            "saving_mean_%": 100.0 * conn["connectivity_saving"].mean(),
            "saving_median_%": 100.0 * conn["connectivity_saving"].median(),
            "fps_reached_n": n_fps,
            "fps_reach_%": 100.0 * n_fps / n if n else np.nan,
            "fps_points_median": fps["fps_points"].median(),
        })

    return pd.DataFrame(rows)


def safe_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon; return NaN when undefined or too few pairs."""
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]

    if len(a) < 3:
        return np.nan, np.nan

    d = b - a
    if np.allclose(d, 0):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def paired_comparison(e2: pd.DataFrame, e3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common_targets = sorted(
        set(e2["random_target"].dropna().astype(int))
        & set(e3["random_target"].dropna().astype(int))
    )

    for target in common_targets:
        a = e2[e2["random_target"] == target].copy()
        b = e3[e3["random_target"] == target].copy()

        m = a.merge(
            b,
            on="object",
            suffixes=("_e2", "_e3"),
            how="inner",
        )

        both = m[
            m["connectivity_points_e2"].notna()
            & m["connectivity_points_e3"].notna()
            ].copy()

        stat_points, p_points = safe_wilcoxon(
            both["connectivity_points_e2"].to_numpy(float),
            both["connectivity_points_e3"].to_numpy(float),
        )
        stat_speed, p_speed = safe_wilcoxon(
            both["connectivity_speedup_e2"].to_numpy(float),
            both["connectivity_speedup_e3"].to_numpy(float),
        )

        rows.append({
            "target": target,
            "common_objects": len(m),
            "both_reached": len(both),
            "both_reached_%": 100.0 * len(both) / len(m) if len(m) else np.nan,
            "E2_points_median": both["connectivity_points_e2"].median(),
            "E3_queries_median": both["connectivity_points_e3"].median(),
            "E2_speedup_median_x": both["connectivity_speedup_e2"].median(),
            "E3_speedup_median_x": both["connectivity_speedup_e3"].median(),
            "E3_minus_E2_speedup_median_x": (
                (both["connectivity_speedup_e3"] - both["connectivity_speedup_e2"]).median()
            ),
            "E3_over_E2_cost_median_x": (
                (both["connectivity_points_e3"] / both["connectivity_points_e2"]).median()
            ),
            "wilcoxon_points_p": p_points,
            "wilcoxon_speedup_p": p_speed,
        })

    return pd.DataFrame(rows)


def object_overlap_table(e2: pd.DataFrame, e3: pd.DataFrame) -> pd.DataFrame:
    o2 = set(e2["object"].astype(str).unique())
    o3 = set(e3["object"].astype(str).unique())
    common = o2 & o3

    return pd.DataFrame([{
        "E2_objects_completed": len(o2),
        "E3_objects_completed": len(o3),
        "common_objects": len(common),
        "E2_only": len(o2 - o3),
        "E3_only": len(o3 - o2),
        "overlap_%_of_E3": 100.0 * len(common) / len(o3) if o3 else np.nan,
    }])


def print_table(title: str, df: pd.DataFrame):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if df.empty:
        print("No rows available yet.")
        return

    with pd.option_context(
            "display.max_columns", None,
            "display.width", 220,
            "display.float_format", lambda x: f"{x:.3f}",
    ):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compare partial or complete E2 and E3 grasp exploration results."
    )
    parser.add_argument(
        "--e2",
        type=Path,
        default=Path("results/online_connectivity/online_connectivity_matches.csv"),
        help="E2 online_connectivity_matches.csv",
    )
    parser.add_argument(
        "--e3",
        type=Path,
        default=Path("results/active_connectivity_e3/online_connectivity_matches.csv"),
        help="E3 online_connectivity_matches.csv",
    )
    parser.add_argument(
        "--expected-objects",
        type=int,
        default=200,
        help="Expected final number of objects; used only to mark partial runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/e2_e3_comparison"),
        help="Directory for comparison CSVs.",
    )
    args = parser.parse_args()

    e2 = load_matches(args.e2, "E2")
    e3 = load_matches(args.e3, "E3")

    n2 = e2["object"].nunique()
    n3 = e3["object"].nunique()
    partial = (n2 < args.expected_objects) or (n3 < args.expected_objects)

    print("\nE2 vs E3 comparison")
    print(f"E2 checkpoint: {args.e2}")
    print(f"E3 checkpoint: {args.e3}")
    print(f"Completed unique objects: E2={n2}, E3={n3}, expected={args.expected_objects}")

    if partial:
        print(
            "STATUS: PARTIAL / INTERIM ANALYSIS. "
            "At least one run is not complete; do not treat p-values or aggregate estimates as final."
        )
    else:
        print("STATUS: both result files contain the expected number of objects.")

    overlap = object_overlap_table(e2, e3)
    summary = pd.concat(
        [experiment_summary(e2, "E2"), experiment_summary(e3, "E3")],
        ignore_index=True,
    )
    paired = paired_comparison(e2, e3)

    print_table("OBJECT OVERLAP", overlap)
    print_table("E2 / E3 DESCRIPTIVE COMPARISON", summary)
    print_table("PAIRED E2 vs E3 ON COMMON OBJECTS", paired)

    print("\nInterpretation notes:")
    print("  • E2 'points' are known-successful grasp observations.")
    print("  • E3 'points' in the source CSV are actually outcome QUERIES.")
    print("  • Reach-rate is reported separately because 'not reached' is censored by the run budget.")
    print("  • Means/medians of points and speedup use REACHED objects only.")
    print("  • Paired Wilcoxon tests use only common objects where BOTH E2 and E3 reached the target.")
    print("    Therefore these tests are descriptive and can be selection-biased when reach rates are low.")
    print("  • While the experiment is unfinished, all tests are interim only.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.out_dir / "object_overlap.csv", index=False)
    summary.to_csv(args.out_dir / "experiment_summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_e2_e3.csv", index=False)

    print(f"\nSaved comparison tables to: {args.out_dir}")


if __name__ == "__main__":
    # Keep warnings visible but avoid failing an interim run because a statistical
    # test is undefined for a very small number of completed paired objects.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()