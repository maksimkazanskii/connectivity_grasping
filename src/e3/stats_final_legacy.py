from pathlib import Path
import argparse

import numpy as np
import pandas as pd


POOL_CONFIG = {
    200: 50,
    400: 100,
    800: 200,
    1600: 400,
}

METHODS = {
    "random": {
        "label": "Random",
        "explore": "random_exploration_successes",
        "exploit": "random_exploitation_successes",
        "auc": "random_exploitation_auc",
    },
    "fps": {
        "label": "FPS",
        "explore": "fps_exploration_successes",
        "exploit": "fps_exploitation_successes",
        "auc": "fps_exploitation_auc",
    },
    "connectivity": {
        "label": "Connectivity-aware",
        "explore": "connectivity_exploration_successes",
        "exploit": "connectivity_exploitation_successes",
        "auc": "connectivity_exploitation_auc",
    },
}


def mean_sd(x):
    """Return mean and sample SD across objects."""
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan, np.nan

    return x.mean(), x.std(ddof=1)


def median_iqr(x):
    """Return median, Q25, and Q75 across objects."""
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    return (
        x.median(),
        x.quantile(0.25),
        x.quantile(0.75),
    )


def bootstrap_mean_ci(x, n_boot=10000, seed=12345):
    """Return mean and percentile-bootstrap 95% CI across objects."""
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        bootstrap_means[i] = sample.mean()

    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return x.mean(), low, high


def check_columns(df, required, path):
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns in {path}:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


def analyze_pool(df, pool_size, query_budget):
    """
    Compute all statistics needed for the E3 LaTeX table.

    Statistics are computed across rows of object_summary.csv,
    i.e. across objects.
    """

    rows = []

    # ------------------------------------------------------------
    # Random reference
    # ------------------------------------------------------------

    random_auc = pd.to_numeric(
        df["random_exploitation_auc"],
        errors="coerce",
    )

    # Maximum possible AUC for this design.
    #
    # Pool B contains K successful and K failed candidates,
    # and Phase 2 makes K queries.
    #
    # Therefore:
    #
    # A_max = K^2 / 2
    #
    # under the trapezoidal AUC definition used in E3.
    #
    amax = query_budget ** 2 / 2.0

    for method, config in METHODS.items():

        explore = pd.to_numeric(
            df[config["explore"]],
            errors="coerce",
        )

        exploit = pd.to_numeric(
            df[config["exploit"]],
            errors="coerce",
        )

        auc = pd.to_numeric(
            df[config["auc"]],
            errors="coerce",
        )

        normalized_auc = auc / amax

        # --------------------------------------------------------
        # Basic object-level statistics
        # --------------------------------------------------------

        explore_mean, explore_sd = mean_sd(explore)
        exploit_mean, exploit_sd = mean_sd(exploit)
        norm_auc_mean, norm_auc_sd = mean_sd(normalized_auc)

        # --------------------------------------------------------
        # Gain relative to Random
        # --------------------------------------------------------

        if method == "random":

            gain_mean = np.nan
            gain_sd = np.nan
            gain_ci_low = np.nan
            gain_ci_high = np.nan
            gain_median = np.nan
            gain_q25 = np.nan
            gain_q75 = np.nan
            win_rate = np.nan
            wins = np.nan

        else:

            valid = (
                    auc.notna()
                    & random_auc.notna()
                    & (random_auc != 0)
            )

            gain = pd.Series(
                np.nan,
                index=df.index,
                dtype=float,
            )

            gain.loc[valid] = (
                    (
                            auc.loc[valid]
                            - random_auc.loc[valid]
                    )
                    / random_auc.loc[valid]
                    * 100.0
            )

            gain_mean, gain_sd = mean_sd(gain)

            # Paired object-level bootstrap: gain is computed from each
            # object's method and Random AUC before resampling objects.
            _, gain_ci_low, gain_ci_high = bootstrap_mean_ci(
                gain,
                n_boot=10000,
                seed=12345 + pool_size + (0 if method == "fps" else 1),
            )

            (
                gain_median,
                gain_q25,
                gain_q75,
            ) = median_iqr(gain)

            comparison_valid = (
                    auc.notna()
                    & random_auc.notna()
            )

            wins = int(
                (
                        auc[comparison_valid]
                        > random_auc[comparison_valid]
                ).sum()
            )

            n_comparisons = int(comparison_valid.sum())

            win_rate = (
                wins / n_comparisons
                if n_comparisons > 0
                else np.nan
            )

        rows.append(
            {
                "pool_size": pool_size,
                "query_budget": query_budget,
                "method": config["label"],
                "objects": len(df),

                "phase1_success_mean": explore_mean,
                "phase1_success_sd": explore_sd,

                "phase2_success_mean": exploit_mean,
                "phase2_success_sd": exploit_sd,

                "normalized_auc_mean": norm_auc_mean,
                "normalized_auc_sd": norm_auc_sd,

                "gain_vs_random_mean_percent": gain_mean,
                "gain_vs_random_sd_percent": gain_sd,
                "gain_vs_random_ci_low_percent": gain_ci_low,
                "gain_vs_random_ci_high_percent": gain_ci_high,

                "gain_vs_random_median_percent": gain_median,
                "gain_vs_random_q25_percent": gain_q25,
                "gain_vs_random_q75_percent": gain_q75,

                "wins_vs_random": wins,
                "win_rate_vs_random": win_rate,
            }
        )

    return rows


def format_mean_sd(mean, sd, decimals=2):
    if pd.isna(mean):
        return "--"

    return (
        f"{mean:.{decimals}f} "
        f"+/- {sd:.{decimals}f}"
    )


def format_median_iqr(median, q25, q75):
    if pd.isna(median):
        return "--"

    return (
        f"{median:+.2f} "
        f"[{q25:+.2f}--{q75:+.2f}]"
    )


def print_table(summary):
    print()
    print("=" * 120)
    print("E3 TABLE STATISTICS")
    print("=" * 120)

    for pool_size in POOL_CONFIG:

        sub = summary[
            summary["pool_size"] == pool_size
            ]

        query_budget = POOL_CONFIG[pool_size]

        print()
        print(
            f"N = {pool_size} | "
            f"K = {query_budget}"
        )
        print("-" * 120)

        for _, row in sub.iterrows():

            phase1 = format_mean_sd(
                row["phase1_success_mean"],
                row["phase1_success_sd"],
                2,
            )

            phase2 = format_mean_sd(
                row["phase2_success_mean"],
                row["phase2_success_sd"],
                2,
            )

            norm_auc = format_mean_sd(
                row["normalized_auc_mean"],
                row["normalized_auc_sd"],
                4,
            )

            if row["method"] == "Random":

                mean_gain = "--"
                median_gain = "--"
                win_rate = "--"

            else:

                mean_gain = (
                    f"{row['gain_vs_random_mean_percent']:+.2f} "
                    f"[{row['gain_vs_random_ci_low_percent']:+.2f}, "
                    f"{row['gain_vs_random_ci_high_percent']:+.2f}]"
                )

                median_gain = format_median_iqr(
                    row[
                        "gain_vs_random_median_percent"
                    ],
                    row[
                        "gain_vs_random_q25_percent"
                    ],
                    row[
                        "gain_vs_random_q75_percent"
                    ],
                )

                win_rate = (
                    f"{100 * row['win_rate_vs_random']:.1f}%"
                )

            print(
                f"{row['method']:20s} | "
                f"P1 {phase1:18s} | "
                f"P2 {phase2:18s} | "
                f"AUC {norm_auc:18s} | "
                f"mean gain {mean_gain:16s} | "
                f"median {median_gain:24s} | "
                f"win {win_rate}"
            )


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

    all_rows = []

    object_sets = {}

    # ============================================================
    # PROCESS ALL FOUR EXPERIMENTS
    # ============================================================

    for pool_size, query_budget in POOL_CONFIG.items():

        pool_dir = base_dir / f"pool_{pool_size}"
        input_path = pool_dir / "object_summary.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Could not find:\n{input_path}"
            )

        print(f"Reading {input_path}")

        df = pd.read_csv(input_path)

        required = []

        for config in METHODS.values():
            required.extend(
                [
                    config["explore"],
                    config["exploit"],
                    config["auc"],
                ]
            )

        check_columns(
            df,
            sorted(set(required)),
            input_path,
        )

        # --------------------------------------------------------
        # Object identity check
        # --------------------------------------------------------

        if "object" in df.columns:
            object_sets[pool_size] = set(
                df["object"].astype(str)
            )

        rows = analyze_pool(
            df,
            pool_size,
            query_budget,
        )

        all_rows.extend(rows)

    # ============================================================
    # CREATE SUMMARY
    # ============================================================

    summary = pd.DataFrame(all_rows)

    output_path = (
            base_dir / "e3_table_statistics.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    # ============================================================
    # PRINT TABLE VALUES
    # ============================================================

    print_table(summary)

    # ============================================================
    # COMMON COHORT CHECK
    # ============================================================

    if len(object_sets) == len(POOL_CONFIG):

        print()
        print("=" * 72)
        print("OBJECT COHORT CHECK")
        print("=" * 72)

        reference_pool = next(iter(POOL_CONFIG))
        reference = object_sets[reference_pool]

        identical = all(
            objects == reference
            for objects in object_sets.values()
        )

        common = set.intersection(
            *object_sets.values()
        )

        for pool_size in POOL_CONFIG:
            print(
                f"N={pool_size:4d}: "
                f"{len(object_sets[pool_size])} objects"
            )

        print(
            f"\nCommon objects: "
            f"{len(common)}"
        )

        print(
            f"Exact same cohort: "
            f"{identical}"
        )

        if not identical:

            print()
            print("WARNING:")
            print(
                "The four experiments do not contain "
                "exactly the same object cohort."
            )

    else:

        print()
        print(
            "Object cohort check skipped because the "
            "'object' column was not available in every file."
        )

    # ============================================================
    # SAVE
    # ============================================================

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)
    print(output_path)


if __name__ == "__main__":
    main()