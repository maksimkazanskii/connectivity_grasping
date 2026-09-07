from pathlib import Path
import argparse

import numpy as np
import pandas as pd


POOL_SIZES = [200, 400, 800, 1600]

METHOD_AUC_COLUMNS = {
    "random": "random_exploitation_auc",
    "fps": "fps_exploitation_auc",
    "connectivity": "connectivity_exploitation_auc",
}


def analyze_scale(df, pool_size):
    """Add derived quantities and return per-scale statistics."""

    df = df.copy()
    df["pool_size"] = pool_size

    # ============================================================
    # DERIVED OBJECT-LEVEL VARIABLES
    # ============================================================

    df["ca_explore_gain"] = (
            df["connectivity_exploration_successes"]
            - df["random_exploration_successes"]
    )

    df["fps_explore_gain"] = (
            df["fps_exploration_successes"]
            - df["random_exploration_successes"]
    )

    df["ca_minus_fps_gain"] = (
            df["ca_gain_vs_random_percent"]
            - df["fps_gain_vs_random_percent"]
    )

    # ============================================================
    # ORACLE POLICY
    # ============================================================

    auc_columns = list(METHOD_AUC_COLUMNS.values())

    df["oracle_auc"] = df[auc_columns].max(axis=1)

    reverse_method_map = {
        column: method
        for method, column in METHOD_AUC_COLUMNS.items()
    }

    df["oracle_method"] = (
        df[auc_columns]
        .idxmax(axis=1)
        .map(reverse_method_map)
    )

    df["oracle_gain_vs_random_percent"] = (
            (df["oracle_auc"] - df["random_exploitation_auc"])
            / df["random_exploitation_auc"]
            * 100.0
    )

    # ============================================================
    # CORRELATIONS
    # ============================================================

    variables = [
        "ca_gain_vs_random_percent",
        "fps_gain_vs_random_percent",
        "ca_minus_fps_gain",
        "ca_explore_gain",
        "fps_explore_gain",
        "connectivity_exploration_successes",
        "fps_exploration_successes",
        "random_exploration_successes",
        "random_exploitation_auc",
    ]

    corr_data = df[variables]

    pearson = corr_data.corr(method="pearson")
    spearman = corr_data.corr(method="spearman")

    # ============================================================
    # SUMMARY
    # ============================================================

    ca_beats_fps = (
            df["connectivity_exploitation_auc"]
            > df["fps_exploitation_auc"]
    )

    fps_beats_ca = (
            df["fps_exploitation_auc"]
            > df["connectivity_exploitation_auc"]
    )

    oracle_gain = df["oracle_gain_vs_random_percent"]

    mean_aucs = {
        method: df[column].mean()
        for method, column in METHOD_AUC_COLUMNS.items()
    }

    best_fixed_method = max(mean_aucs, key=mean_aucs.get)
    best_fixed_auc = mean_aucs[best_fixed_method]

    mean_oracle_auc = df["oracle_auc"].mean()

    oracle_vs_best_fixed = (
            (mean_oracle_auc - best_fixed_auc)
            / best_fixed_auc
            * 100.0
    )

    oracle_counts = df["oracle_method"].value_counts()

    summary = {
        "pool_size": pool_size,
        "objects": len(df),

        # Mean AUCs
        "random_mean_auc": mean_aucs["random"],
        "fps_mean_auc": mean_aucs["fps"],
        "ca_mean_auc": mean_aucs["connectivity"],
        "oracle_mean_auc": mean_oracle_auc,

        # Best fixed
        "best_fixed_method": best_fixed_method,
        "best_fixed_auc": best_fixed_auc,
        "oracle_vs_best_fixed_percent": oracle_vs_best_fixed,

        # Existing gain statistics
        "fps_mean_gain_vs_random_percent":
            df["fps_gain_vs_random_percent"].mean(),
        "fps_median_gain_vs_random_percent":
            df["fps_gain_vs_random_percent"].median(),
        "ca_mean_gain_vs_random_percent":
            df["ca_gain_vs_random_percent"].mean(),
        "ca_median_gain_vs_random_percent":
            df["ca_gain_vs_random_percent"].median(),

        # CA vs FPS
        "fraction_ca_beats_fps": ca_beats_fps.mean(),
        "fraction_fps_beats_ca": fps_beats_ca.mean(),

        # Oracle
        "oracle_mean_gain_vs_random_percent": oracle_gain.mean(),
        "oracle_median_gain_vs_random_percent": oracle_gain.median(),
        "oracle_q25_gain_vs_random_percent": oracle_gain.quantile(0.25),
        "oracle_q75_gain_vs_random_percent": oracle_gain.quantile(0.75),

        "oracle_random_count": int(
            oracle_counts.get("random", 0)
        ),
        "oracle_fps_count": int(
            oracle_counts.get("fps", 0)
        ),
        "oracle_ca_count": int(
            oracle_counts.get("connectivity", 0)
        ),

        "oracle_random_fraction":
            oracle_counts.get("random", 0) / len(df),
        "oracle_fps_fraction":
            oracle_counts.get("fps", 0) / len(df),
        "oracle_ca_fraction":
            oracle_counts.get("connectivity", 0) / len(df),

        # Key correlations
        "pearson_ca_fps_gain":
            pearson.loc[
                "ca_gain_vs_random_percent",
                "fps_gain_vs_random_percent",
            ],

        "spearman_ca_fps_gain":
            spearman.loc[
                "ca_gain_vs_random_percent",
                "fps_gain_vs_random_percent",
            ],

        "pearson_ca_gain_vs_ca_explore_gain":
            pearson.loc[
                "ca_gain_vs_random_percent",
                "ca_explore_gain",
            ],

        "spearman_ca_gain_vs_ca_explore_gain":
            spearman.loc[
                "ca_gain_vs_random_percent",
                "ca_explore_gain",
            ],
    }

    return df, pearson, spearman, summary


def cross_scale_correlations(all_df, value_column):
    """
    Correlation of object-level values between pool sizes.

    Requires the same object IDs to occur at each scale.
    """

    pivot = all_df.pivot(
        index="object",
        columns="pool_size",
        values=value_column,
    )

    pearson = pivot.corr(method="pearson")
    spearman = pivot.corr(method="spearman")

    return pivot, pearson, spearman


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base_results_dir",
        nargs="?",
        default="results/e3_disjoint_pools_results",
        help=(
            "Directory containing pool_200, pool_400, "
            "pool_800, and pool_1600"
        ),
    )

    args = parser.parse_args()

    base_dir = Path(args.base_results_dir)

    # ============================================================
    # LOAD + ANALYZE ALL SCALES
    # ============================================================

    all_frames = []
    summaries = []
    object_sets = {}

    print()
    print("=" * 78)
    print("E3 MULTI-SCALE META-ANALYSIS")
    print("=" * 78)

    for pool_size in POOL_SIZES:

        results_dir = base_dir / f"pool_{pool_size}"
        input_path = results_dir / "object_summary.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Could not find {input_path}"
            )

        df = pd.read_csv(input_path)

        if "object" not in df.columns:
            raise ValueError(
                f"{input_path} does not contain an 'object' column."
            )

        object_sets[pool_size] = set(df["object"])

        df, pearson, spearman, summary = analyze_scale(
            df,
            pool_size,
        )

        # Save scale-specific enriched outputs
        df.to_csv(
            results_dir / "meta_object_summary.csv",
            index=False,
            )

        pearson.to_csv(
            results_dir / "gain_correlations_pearson.csv"
        )

        spearman.to_csv(
            results_dir / "gain_correlations_spearman.csv"
        )

        all_frames.append(df)
        summaries.append(summary)

        # --------------------------------------------------------
        # SCALE REPORT
        # --------------------------------------------------------

        print()
        print("-" * 78)
        print(f"POOL SIZE N = {pool_size}")
        print("-" * 78)

        print(f"Objects: {len(df)}")

        print("\nMean Phase-2 AUC:")
        print(
            f"  Random: {summary['random_mean_auc']:.2f}"
        )
        print(
            f"  FPS:    {summary['fps_mean_auc']:.2f}"
        )
        print(
            f"  CA:     {summary['ca_mean_auc']:.2f}"
        )
        print(
            f"  Oracle: {summary['oracle_mean_auc']:.2f}"
        )

        print("\nGain vs Random:")
        print(
            f"  FPS mean:   "
            f"{summary['fps_mean_gain_vs_random_percent']:+.3f}%"
        )
        print(
            f"  FPS median: "
            f"{summary['fps_median_gain_vs_random_percent']:+.3f}%"
        )
        print(
            f"  CA mean:    "
            f"{summary['ca_mean_gain_vs_random_percent']:+.3f}%"
        )
        print(
            f"  CA median:  "
            f"{summary['ca_median_gain_vs_random_percent']:+.3f}%"
        )

        print("\nCA vs FPS:")
        print(
            f"  CA better:  "
            f"{summary['fraction_ca_beats_fps']:.3f}"
        )
        print(
            f"  FPS better: "
            f"{summary['fraction_fps_beats_ca']:.3f}"
        )

        print("\nCA gain vs FPS gain:")
        print(
            f"  Pearson r  = "
            f"{summary['pearson_ca_fps_gain']:+.4f}"
        )
        print(
            f"  Spearman ρ = "
            f"{summary['spearman_ca_fps_gain']:+.4f}"
        )

        print("\nOracle method frequencies:")
        print(
            f"  Random: "
            f"{summary['oracle_random_count']:3d}/{len(df)} "
            f"({summary['oracle_random_fraction']:.3f})"
        )
        print(
            f"  FPS:    "
            f"{summary['oracle_fps_count']:3d}/{len(df)} "
            f"({summary['oracle_fps_fraction']:.3f})"
        )
        print(
            f"  CA:     "
            f"{summary['oracle_ca_count']:3d}/{len(df)} "
            f"({summary['oracle_ca_fraction']:.3f})"
        )

        print("\nMeta-learning headroom:")
        print(
            f"  Best fixed method:     "
            f"{summary['best_fixed_method']}"
        )
        print(
            f"  Best fixed mean AUC:   "
            f"{summary['best_fixed_auc']:.2f}"
        )
        print(
            f"  Oracle mean AUC:       "
            f"{summary['oracle_mean_auc']:.2f}"
        )
        print(
            f"  Oracle vs best fixed:  "
            f"{summary['oracle_vs_best_fixed_percent']:+.3f}%"
        )

    # ============================================================
    # COMBINE ALL SCALES
    # ============================================================

    all_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summaries)

    all_df.to_csv(
        base_dir / "meta_all_scales_object_summary.csv",
        index=False,
        )

    summary_df.to_csv(
        base_dir / "meta_scale_summary.csv",
        index=False,
        )

    # ============================================================
    # CHECK COMMON OBJECT COHORT
    # ============================================================

    print()
    print("=" * 78)
    print("COMMON-COHORT CHECK")
    print("=" * 78)

    reference_size = POOL_SIZES[0]
    reference_objects = object_sets[reference_size]

    common_objects = set.intersection(
        *object_sets.values()
    )

    all_identical = all(
        object_sets[n] == reference_objects
        for n in POOL_SIZES
    )

    for pool_size in POOL_SIZES:
        print(
            f"N={pool_size:4d}: "
            f"{len(object_sets[pool_size])} objects"
        )

    print(
        f"\nObjects common to all scales: "
        f"{len(common_objects)}"
    )

    print(
        f"Exact same cohort at all scales: "
        f"{all_identical}"
    )

    # ============================================================
    # CROSS-SCALE CORRELATIONS
    # ============================================================

    print()
    print("=" * 78)
    print("CROSS-SCALE OBJECT CONSISTENCY")
    print("=" * 78)

    cross_scale_variables = [
        "ca_gain_vs_random_percent",
        "fps_gain_vs_random_percent",
        "ca_minus_fps_gain",
    ]

    for variable in cross_scale_variables:

        pivot, pearson, spearman = cross_scale_correlations(
            all_df,
            variable,
        )

        pivot.to_csv(
            base_dir /
            f"cross_scale_{variable}.csv"
        )

        pearson.to_csv(
            base_dir /
            f"cross_scale_{variable}_pearson.csv"
        )

        spearman.to_csv(
            base_dir /
            f"cross_scale_{variable}_spearman.csv"
        )

        print()
        print(variable)

        print("  Spearman correlations:")

        for i, n1 in enumerate(POOL_SIZES):
            for n2 in POOL_SIZES[i + 1:]:
                value = spearman.loc[n1, n2]

                print(
                    f"    N={n1:4d} vs N={n2:4d}: "
                    f"{value:+.4f}"
                )

    # ============================================================
    # ORACLE METHOD STABILITY ACROSS SCALES
    # ============================================================

    oracle_pivot = all_df.pivot(
        index="object",
        columns="pool_size",
        values="oracle_method",
    )

    oracle_pivot.to_csv(
        base_dir / "oracle_method_by_scale.csv"
    )

    print()
    print("=" * 78)
    print("ORACLE STRATEGY STABILITY")
    print("=" * 78)

    for i, n1 in enumerate(POOL_SIZES):
        for n2 in POOL_SIZES[i + 1:]:

            pair = oracle_pivot[[n1, n2]].dropna()

            agreement = (
                    pair[n1] == pair[n2]
            ).mean()

            print(
                f"N={n1:4d} vs N={n2:4d}: "
                f"same best method = {agreement:.3f} "
                f"({(pair[n1] == pair[n2]).sum()}/{len(pair)})"
            )

    # ============================================================
    # COMPACT SCALE TABLE
    # ============================================================

    print()
    print("=" * 78)
    print("SCALE SUMMARY")
    print("=" * 78)

    display_columns = [
        "pool_size",
        "fps_mean_gain_vs_random_percent",
        "ca_mean_gain_vs_random_percent",
        "oracle_random_fraction",
        "oracle_fps_fraction",
        "oracle_ca_fraction",
        "oracle_vs_best_fixed_percent",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved combined outputs:")
    print(
        f"  {base_dir / 'meta_all_scales_object_summary.csv'}"
    )
    print(
        f"  {base_dir / 'meta_scale_summary.csv'}"
    )
    print(
        f"  {base_dir / 'oracle_method_by_scale.csv'}"
    )
    print(
        f"  {base_dir / 'cross_scale_*'}"
    )


if __name__ == "__main__":
    main()
