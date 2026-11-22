#!/usr/bin/env python3
"""E4 prototype-count ablation: seed-level heatmaps and target-level violins.

Requires: numpy, pandas, matplotlib. CSV schema is auto-detected for common
long-format (method + success/auc) and wide-format (method_success/auc) outputs.
Use --inspect to print columns when the experiment uses a different schema.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm

PROTOTYPES = (4, 8, 16, 32, 64)
METRICS = {
    'success': ('success_at_budget', 'success_rate', 'success', 'successes', 'success_percent', 'success_pct', 'found'),
    'auc': ('cumulative_success_auc', 'auc', 'success_auc', 'exploration_auc'),
}
METHOD_COLS = ('method', 'algorithm', 'policy', 'strategy')
SEED_COLS = ('seed', 'random_seed')
TARGET_COLS = ('target_id', 'target', 'target_object', 'object_id', 'object', 'episode', 'episode_id')
LABELS = {
    'random': 'Random', 'connectivity_encoder': 'Connectivity encoder',
    'fps_geometry_encoder': 'FPS geometry', 'geometry_encoder': 'Geometry encoder',
    'dynamic_geometry_encoder': 'Dynamic geometry',
    'dynamic_fps_geometry_encoder': 'Dynamic FPS geometry',
    'dynamic_random_geometry_encoder': 'Dynamic random geometry',
    'target_ca': 'Target CA',
}
# Steelblue -> neutral ivory -> orange, with blue-red violins unchanged.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "steelblue_orange", ["#4682B4", "#94B9D2", "#F4F1EB", "#F4B183", "#E47B32"], N=256
)
VIOLIN_CMAP = 'RdBu_r'

# Shared algorithm colors: identical to the paired-AUC violin plot.
METHOD_COLORS = {
    'connectivity_encoder': '#2868A5',
    'dynamic_fps_geometry_encoder': '#72B4D6',
    'dynamic_geometry_encoder': '#648A92',
    'dynamic_random_geometry_encoder': '#EF956B',
    'target_ca': '#B51F36',
    'random': '#777777',
}


# Compact publication labels; original method IDs and statistical data are unchanged.
VIOLIN_LABELS = {
    'connectivity_encoder': 'Connectivity',
    'dynamic_fps_geometry_encoder': 'Dynamic FPS',
    'dynamic_geometry_encoder': 'Dynamic geometry',
    'dynamic_random_geometry_encoder': 'Dynamic random',
    'target_ca': 'Target CA',
    'random': 'Random',
}

def short_violin_label(method):
    return VIOLIN_LABELS.get(method, pretty(method))

def find_col(columns, choices):
    lookup = {str(c).lower(): c for c in columns}
    return next((lookup[c] for c in choices if c in lookup), None)

def metric_col(columns, metric):
    return find_col(columns, METRICS[metric])

def tidy_csv(path, seed, prototypes, metric):
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f'{path}: empty CSV')
    mcol = find_col(raw.columns, METHOD_COLS)
    scol = find_col(raw.columns, SEED_COLS)
    tcol = find_col(raw.columns, TARGET_COLS)
    valcol = metric_col(raw.columns, metric)
    # E4 raw trajectory: 100 steps for each (target, method).
    # Reduce BEFORE concatenating files to avoid millions of duplicate target keys.
    if {'step', 'success', 'cumulative_success'}.issubset(raw.columns) and mcol and tcol:
        if scol is not None and not (raw[scol] == seed).all():
            raise ValueError(f'{path}: seed values do not match directory seed={seed}')
        raw['step'] = pd.to_numeric(raw['step'], errors='raise')
        raw['cumulative_success'] = pd.to_numeric(raw['cumulative_success'], errors='raise')
        groups = raw.groupby([mcol, tcol], sort=False, observed=True)
        checks = groups.agg(n=('step', 'size'), unique_steps=('step', 'nunique'),
                            last_step=('step', 'max'), first_step=('step', 'min'))
        bad = checks[(checks.n != 100) | (checks.unique_steps != 100) |
                     (checks.first_step != 1) | (checks.last_step != 100)]
        if len(bad):
            raise ValueError(f'{path}: {len(bad)} incomplete/duplicate trajectories; '
                             'expected steps 1..100 per (method, target)')
        if metric == 'success':
            final = raw.loc[raw['step'].eq(100), [mcol, tcol, 'cumulative_success']].copy()
            final['value'] = final['cumulative_success']  # percent, budget=100
        else:
            final = groups['cumulative_success'].mean().reset_index(name='value')
            # Mean cumulative successes over steps 1..100: AUC on 0..100 scale.
        out = pd.DataFrame({'method': final[mcol].astype(str),
                            'target': final[tcol].astype(str),
                            'value': pd.to_numeric(final['value'], errors='raise')})
    elif mcol is not None and valcol is not None:
        out = pd.DataFrame({'method': raw[mcol].astype(str),
                            'value': pd.to_numeric(raw[valcol], errors='coerce')})
        out['target'] = raw[tcol].astype(str) if tcol else out.groupby('method').cumcount().astype(str)
    else:
        suffixes = METRICS[metric]
        found = []
        for col in raw.columns:
            low = str(col).lower()
            for suffix in sorted(suffixes, key=len, reverse=True):
                if low.endswith('_' + suffix):
                    name = str(col)[:-(len(suffix) + 1)]
                    if name and name.lower() not in ('mean', 'std', 'sd'):
                        found.append((col, name))
                    break
        if not found:
            raise ValueError(f'{path}: cannot identify per-target {metric} values. Columns: {list(raw.columns)}')
        id_vars = [c for c in (scol, tcol) if c is not None]
        melted = raw[id_vars + [c for c, _ in found]].melt(id_vars=id_vars, var_name='_column', value_name='value')
        out = pd.DataFrame({'method': melted['_column'].map(dict(found)),
                            'target': melted[tcol].astype(str) if tcol else melted.groupby('_column').cumcount().astype(str),
                            'value': pd.to_numeric(melted['value'], errors='coerce')})
    out['seed'] = seed
    out['prototypes'] = prototypes
    out = out[['seed', 'prototypes', 'method', 'target', 'value']]
    out = out[np.isfinite(out['value'])].copy()
    if out.empty:
        raise ValueError(f'{path}: no finite per-target observations')
    return out

def load(base, metric, allow_partial):
    frames = []
    missing = []
    for seed in range(5):
        for n in PROTOTYPES:
            path = base / f'seed_{seed}' / f'prototypes_{n}.csv'
            if path.is_file():
                frames.append(tidy_csv(path, seed, n, metric))
            else:
                missing.append(str(path))
    if missing and not allow_partial:
        raise FileNotFoundError(f'Missing {len(missing)} experiment files (use --allow-partial to plot completed runs):\n' + '\n'.join(missing))
    if not frames:
        raise FileNotFoundError(f'No experiment CSV files found under {base}')
    if missing:
        print(f'WARNING: {len(missing)} missing CSV files; partial results are not final.')
    df = pd.concat(frames, ignore_index=True)
    # Catch accidentally loaded aggregate rows or duplicate targets.
    duplicates = df.duplicated(['seed', 'prototypes', 'method', 'target'])
    if duplicates.any():
        raise ValueError(f'{duplicates.sum()} duplicate (seed, prototypes, method, target) records; check input CSV schema.')
    return df

def summaries(df):
    # Each seed is one independent experimental replicate.
    per_seed = df.groupby(['prototypes', 'method', 'seed'], as_index=False).agg(mean=('value', 'mean'), n_targets=('value', 'size'))
    summary = per_seed.groupby(['prototypes', 'method'], as_index=False).agg(mean=('mean', 'mean'), sd=('mean', lambda x: x.std(ddof=1) if len(x) > 1 else np.nan), n_seeds=('seed', 'size'), min_targets=('n_targets', 'min'), max_targets=('n_targets', 'max'))
    return per_seed, summary

def pretty(method):
    return LABELS.get(method, method.replace('_', ' ').title())

def save(fig, out, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def heatmap(summary, out, metric, center):
    methods = sorted(summary.method.unique())
    mat = summary.pivot(index='method', columns='prototypes', values='mean').reindex(index=methods, columns=PROTOTYPES)
    sd = summary.pivot(index='method', columns='prototypes', values='sd').reindex(index=methods, columns=PROTOTYPES)
    vals = mat.to_numpy(dtype=float)
    valid = vals[np.isfinite(vals)]
    if not len(valid):
        return
    lo, hi = float(valid.min()), float(valid.max())
    if lo == hi:
        lo -= 1; hi += 1
    norm = Normalize(vmin=lo, vmax=hi)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.55 * len(PROTOTYPES) + 2.5), max(3.3, 0.62 * len(methods) + 1.8)))
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad('#f0f0f0')
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    for i in range(len(methods)):
        for j in range(len(PROTOTYPES)):
            mean, sigma = vals[i, j], sd.iloc[i, j]
            if not np.isfinite(mean):
                label = '—'
            elif np.isfinite(sigma):
                label = f'{mean:.2f}\n± {sigma:.2f}'
            else:
                label = f'{mean:.2f}\n± n/a'
            rgba = im.cmap(im.norm(mean)) if np.isfinite(mean) else (1,1,1,1)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, label, ha='center', va='center', fontsize=9, fontweight='semibold', color='white' if luminance < .52 else '#172338')
    ax.set_xticks(range(len(PROTOTYPES)), [str(x) for x in PROTOTYPES])
    ax.set_yticks(range(len(methods)), [pretty(x) for x in methods])
    ax.set_xlabel('Prototypes per reference object', fontsize=11)
    ax.set_ylabel('Algorithm', fontsize=11)
    ax.set_xticks(np.arange(-.5, len(PROTOTYPES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cb.set_label('Mean ' + ('success (%)' if metric == 'success' else 'cumulative success AUC'))
    fig.tight_layout()
    save(fig, out, f'e4_{metric}_prototype_heatmap')

def lineplot(summary, out, metric):
    """Additional prototype-scaling plot: five-seed means with ±1 SD bands."""
    methods = sorted(summary.method.unique())
    if not np.isfinite(summary['mean'].to_numpy(dtype=float)).any():
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    for idx, method in enumerate(methods):
        sub = summary[summary.method == method].sort_values('prototypes')
        x = sub.prototypes.to_numpy(dtype=float)
        y = sub['mean'].to_numpy(dtype=float)
        sd = sub.sd.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y, sd = x[valid], y[valid], sd[valid]
        if not len(x):
            continue
        color = METHOD_COLORS.get(method, '#555555')
        ax.plot(x, y, marker=markers[idx % len(markers)], color=color,
                linewidth=2.2, markersize=6.2, label=pretty(method), zorder=3)
        band = np.isfinite(sd)
        if band.any():
            ax.fill_between(x[band], y[band] - sd[band], y[band] + sd[band],
                            color=color, alpha=.18, linewidth=0, zorder=2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(PROTOTYPES)
    ax.set_xticklabels([str(p) for p in PROTOTYPES])
    ax.set_xlabel('Prototypes per reference object')
    ax.set_ylabel('Success@100 (%)' if metric == 'success' else 'Cumulative success AUC')
    ax.grid(axis='y', linestyle='--', linewidth=.7, alpha=.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='best', ncol=2)
    fig.tight_layout()
    save(fig, out, f'e4_{metric}_prototype_lines_sd')


def violin(df, out, metric, n, selected_seed):
    subset = df[df.prototypes == n]
    if subset.empty:
        print(f'WARNING: no data at {n} prototypes; skipping violin.')
        return
    available = sorted(subset.seed.unique())
    if selected_seed not in available:
        print(f'WARNING: seed {selected_seed} unavailable at {n} prototypes; skipping violin.')
        return
    subset = subset[subset.seed == selected_seed]
    methods = sorted(subset.method.unique())
    arrays = [subset.loc[subset.method == m, 'value'].to_numpy() for m in methods]
    counts = [len(a) for a in arrays]
    if any(count != 500 for count in counts):
        print(f'WARNING: violin seed {selected_seed} contains target counts {dict(zip(methods, counts))}; not all are 500.')
    fig, ax = plt.subplots(figsize=(max(7.0, len(methods) * 1.35 + 1.8), 4.8))
    positions = np.arange(1, len(methods) + 1)
    palette = plt.get_cmap(VIOLIN_CMAP)(np.linspace(.10, .90, max(len(methods), 2)))
    rng = np.random.default_rng(12345)
    for i, (a, color) in enumerate(zip(arrays, palette), start=1):
        if len(a) >= 2 and np.ptp(a) > 0:
            parts = ax.violinplot([a], positions=[i], widths=.78, showmeans=False, showmedians=False, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.43); body.set_linewidth(1.0)
        jitter = rng.normal(0, .055, len(a))
        ax.scatter(i + jitter, a, s=9, alpha=.42, color=color, edgecolors='none', zorder=2, rasterized=True)
        q25, median, q75 = np.percentile(a, [25, 50, 75])
        ax.vlines(i, q25, q75, color=color, lw=4, zorder=3)
        ax.scatter(i, median, s=48, color=color, edgecolors='white', linewidth=.8, zorder=4)
    ax.set_xticks(positions, [short_violin_label(m) for m in methods], rotation=12, ha='right')
    ax.set_ylabel('Success (%)' if metric == 'success' else 'Cumulative success AUC')
    ax.grid(axis='y', linestyle='--', linewidth=.6, alpha=.30, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save(fig, out, f'e4_{metric}_violin_p{n}_seed{selected_seed}')

def violin_auc_gain_vs_random(df, out, n, selected_seed, gain_mode='absolute'):
    """Paired target-level AUC differences; absolute avoids unstable denominators."""
    subset = df[(df.prototypes == n) & (df.seed == selected_seed)]
    if subset.empty:
        print(f'WARNING: no data for {n} prototypes, seed {selected_seed}; skipping gain violin.')
        return
    pivot = subset.pivot(index='target', columns='method', values='value')
    if 'random' not in pivot.columns:
        raise ValueError('Cannot compute paired gain: random baseline is missing.')
    methods = [m for m in sorted(pivot.columns) if m != 'random']
    baseline = pivot['random']
    gains = pd.DataFrame(index=pivot.index)
    for method in methods:
        if gain_mode == 'absolute':
            gains[method] = pivot[method] - baseline
        else:
            # This replicates E3 percentage gains, but is highly unstable when
            # Random AUC is close to zero. Never silently clip or trim values.
            gains[method] = 100 * (pivot[method] - baseline) / baseline.replace(0, np.nan)
    value_col = 'auc_difference' if gain_mode == 'absolute' else 'gain_percent'
    long = gains.reset_index().melt(id_vars='target', var_name='method', value_name=value_col)
    long.insert(0, 'seed', selected_seed)
    long.insert(1, 'prototypes', n)
    stem = f'e4_auc_{gain_mode}_vs_random_p{n}_seed{selected_seed}'
    long.to_csv(out / f'{stem}.csv', index=False)
    arrays = [gains[m].dropna().to_numpy(dtype=float) for m in methods]
    if any(len(a) < 2 for a in arrays):
        raise ValueError('Insufficient paired observations for gain violins.')
    print(f'\nPaired AUC {gain_mode} differences over Random:')
    for method, a in zip(methods, arrays):
        print(f'  {pretty(method):30s} n={len(a):3d} mean={a.mean():+.2f} median={np.median(a):+.2f}')
    fig, ax = plt.subplots(figsize=(max(7.0, len(methods) * 1.48 + 1.8), 4.8))
    positions = np.arange(1, len(methods) + 1)
    colors = ['#2868A5', '#72B4D6', '#648A92', '#EF956B', '#B51F36'][:len(methods)]
    rng = np.random.default_rng(12345)
    for i, (a, color) in enumerate(zip(arrays, colors), start=1):
        if np.ptp(a) > 0:
            parts = ax.violinplot([a], positions=[i], widths=.78,
                                  showmeans=False, showmedians=False, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.43)
                body.set_linewidth(1.0)
        ax.scatter(i + rng.normal(0, .055, len(a)), a, s=9, alpha=.42,
                   color=color, edgecolors='none', zorder=2, rasterized=True)
        q25, median, q75 = np.percentile(a, [25, 50, 75])
        ax.vlines(i, q25, q75, color=color, lw=4, zorder=3)
        ax.scatter(i, median, s=48, color=color, edgecolors='white', linewidth=.8, zorder=4)
    ax.axhline(0, color='0.35', linestyle='--', linewidth=1.2, alpha=.8, zorder=1)
    ax.set_xticks(positions, [short_violin_label(m) for m in methods],
                  rotation=12, ha='right')
    ax.set_ylabel('Paired AUC difference vs Random' if gain_mode == 'absolute'
                  else 'Target-level AUC gain over Random (%)')
    ax.grid(axis='y', linestyle='--', linewidth=.6, alpha=.30, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save(fig, out, f'{stem}_violin')



def holm_adjust(pvalues):
    """Holm step-down familywise correction, preserving missing comparisons."""
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(len(arr), np.nan)
    valid = np.flatnonzero(np.isfinite(arr))
    if not len(valid):
        return out
    order = valid[np.argsort(arr[valid])]
    adjusted = np.maximum.accumulate(np.minimum(1., arr[order] * (len(order) - np.arange(len(order)))))
    out[order] = adjusted
    return out


def paired_seed_statistics(df, metric):
    """Paired target differences are averaged WITHIN seed; seeds are test units.

    One Holm family is all non-connectivity comparisons within a given
    prototype capacity and metric. This does not correct across capacities
    or across the two metrics; these are separate reported families.
    """
    reference = 'connectivity_encoder'
    rows, paired_seed_rows = [], []
    for capacity in sorted(df.prototypes.unique()):
        part = df[df.prototypes.eq(capacity)]
        methods = sorted(set(part.method.unique()) - {reference})
        for baseline in methods:
            seed_diffs, seed_conn, seed_base, counts = [], [], [], []
            for seed in sorted(part.seed.unique()):
                subset = part[part.seed.eq(seed)]
                pair = subset[subset.method.isin([reference, baseline])].pivot(
                    index='target', columns='method', values='value')
                if reference not in pair or baseline not in pair:
                    continue
                pair = pair[[reference, baseline]].dropna()
                if pair.empty:
                    continue
                d = float((pair[reference] - pair[baseline]).mean())
                seed_diffs.append(d)
                seed_conn.append(float(pair[reference].mean()))
                seed_base.append(float(pair[baseline].mean()))
                counts.append(len(pair))
                paired_seed_rows.append(dict(metric=metric, prototypes=capacity,
                                             baseline=baseline, seed=int(seed),
                                             paired_targets=len(pair),
                                             connectivity_mean=seed_conn[-1],
                                             baseline_mean=seed_base[-1], difference=d))
            n = len(seed_diffs)
            if not n:
                continue
            diffs = np.asarray(seed_diffs)
            mean = float(diffs.mean())
            sd = float(diffs.std(ddof=1)) if n > 1 else np.nan
            if n >= 2 and sd > 0:
                se = sd / np.sqrt(n)
                critical = stats.t.ppf(.975, n-1)
                ci_low, ci_high = mean - critical*se, mean + critical*se
                pvalue = float(stats.ttest_1samp(diffs, 0).pvalue)
            elif n >= 2 and sd == 0:
                ci_low = ci_high = mean
                pvalue = 1.0 if mean == 0 else 0.0
            else:
                ci_low = ci_high = pvalue = np.nan
            rows.append(dict(metric=metric, prototypes=int(capacity), baseline=baseline,
                             n_seeds=n, min_paired_targets=min(counts),
                             max_paired_targets=max(counts),
                             connectivity_mean=float(np.mean(seed_conn)),
                             connectivity_sd=float(np.std(seed_conn, ddof=1)) if n>1 else np.nan,
                             baseline_mean=float(np.mean(seed_base)),
                             baseline_sd=float(np.std(seed_base, ddof=1)) if n>1 else np.nan,
                             delta=mean, delta_sd=sd, ci_low=ci_low, ci_high=ci_high,
                             p_raw=pvalue))
    result = pd.DataFrame(rows)
    if not result.empty:
        result['p_holm'] = np.nan
        for (_, _), indexes in result.groupby(['metric', 'prototypes']).groups.items():
            result.loc[indexes, 'p_holm'] = holm_adjust(result.loc[indexes, 'p_raw'])
    return result, pd.DataFrame(paired_seed_rows)


def pformat(value):
    if not np.isfinite(value):
        return '--'
    if value < 1e-4:
        return r'$<10^{-4}$'
    return f'{value:.4f}'


def latex_table(rows, caption, label):
    """Return a standalone LaTeX table* snippet with manuscript-ready numbers."""
    def esc(s):
        return str(s).replace('_', r'\_').replace('&', r'\&')
    def num(x):
        return f'{x:.2f}' if np.isfinite(x) else '--'
    out = [r'\begin{table*}[t]', r'\centering',
           r'\caption{' + caption + '}', r'\label{' + label + '}',
           r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{clcccccc}', r'\hline',
           r'$P$ & Comparison & Connectivity & Baseline & $\Delta$ & 95\% CI for $\Delta$ & $p$ & $p_{\mathrm{Holm}}$ \\',
           r'\hline']
    for _, r in rows.sort_values(['prototypes', 'baseline']).iterrows():
        baseline = esc(pretty(r['baseline']))
        line = (f"{int(r['prototypes'])} & {baseline} & "
                f"${num(r['connectivity_mean'])}\\pm {num(r['connectivity_sd'])}$ & "
                f"${num(r['baseline_mean'])}\\pm {num(r['baseline_sd'])}$ & "
                f"${r['delta']:+.2f}$ & "
                f"$[{num(r['ci_low'])}, {num(r['ci_high'])}]$ & "
                f"{pformat(r['p_raw'])} & {pformat(r['p_holm'])} " + r"\\")
        out.append(line)
    out += [r'\hline', r'\end{tabular}',
            r'\end{table*}']
    return '\n'.join(out) + '\n'


def write_statistics(base, out, allow_partial, main_prototypes):
    combined = []
    for metric in ('success', 'auc'):
        df = load(base, metric, allow_partial)
        results, per_seed = paired_seed_statistics(df, metric)
        if results.empty:
            raise ValueError(f'No paired comparisons available for {metric}')
        results.to_csv(out / f'e4_{metric}_paired_statistics.csv', index=False)
        per_seed.to_csv(out / f'e4_{metric}_paired_seed_differences.csv', index=False)
        caption = ('Success@100' if metric == 'success' else 'Cumulative-success AUC')
        caption += (r' for the E4 prototype-capacity ablation. Values are mean $\pm$ '
                    r'SD across independently trained seeds. $\Delta$ denotes '
                    'connectivity minus baseline; confidence intervals and two-sided '
                    'paired $t$-tests use seed-level mean differences. Holm correction '
                    'is applied across baselines separately for each prototype capacity.')
        (out / f'e4_{metric}_full_table.tex').write_text(
            latex_table(results, caption, f'tab:e4_{metric}_prototype_stats'))
        main = results[results.prototypes.eq(main_prototypes)]
        if main.empty:
            raise ValueError(f'No results for --main-prototypes {main_prototypes}')
        (out / f'e4_{metric}_main_table_p{main_prototypes}.tex').write_text(
            latex_table(main, caption, f'tab:e4_{metric}_main_stats'))
        combined.append(results)
        print(f'\n{metric.upper()} paired seed-level statistics:')
        print(results[['prototypes','baseline','n_seeds','delta','ci_low','ci_high','p_raw','p_holm']].to_string(index=False))
    pd.concat(combined, ignore_index=True).to_csv(out / 'e4_all_paired_statistics.csv', index=False)
    print('\nIMPORTANT: Holm correction is within each (metric, capacity) family, not across all tables.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_dir', nargs='?', type=Path, default=Path('results/e4_prototype_ablation_ref50_dynamic_geometry'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--metric', choices=('success', 'auc'), default='success')
    parser.add_argument('--violin-prototypes', type=int, default=16)
    parser.add_argument('--violin-seed', type=int, default=0, help='One seed gives 500 targets per algorithm; default 0')
    parser.add_argument('--center', type=float, default=None, help='Optional fixed heatmap color center')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--auc-gain-vs-random', action='store_true', help='Plot paired target-level AUC differences over Random')
    parser.add_argument('--gain-mode', choices=('absolute', 'percent'), default='absolute', help='absolute (recommended) or E3-style percent (unstable near zero)')
    parser.add_argument('--no-statistics', action='store_true', help='Disable paired statistical tables')
    parser.add_argument('--main-prototypes', type=int, default=16, choices=PROTOTYPES, help='Prototype count for concise manuscript tables')
    parser.add_argument('--inspect', action='store_true', help='Print CSV columns and exit')
    args = parser.parse_args()
    if args.inspect:
        paths = sorted(args.base_dir.glob('seed_*/prototypes_*.csv'))
        if not paths: raise FileNotFoundError(f'No CSVs in {args.base_dir}')
        for path in paths[:3]: print(path, list(pd.read_csv(path, nrows=2).columns))
        return
    out = args.output_dir or args.base_dir / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    # For the paired-AUC figure, keep the heatmap on ordinary final success.
    heatmap_metric = 'success' if args.auc_gain_vs_random else args.metric
    df = load(args.base_dir, heatmap_metric, args.allow_partial)
    per_seed, summary = summaries(df)
    df.to_csv(out / f'e4_{heatmap_metric}_target_values.csv', index=False)
    per_seed.to_csv(out / f'e4_{heatmap_metric}_per_seed.csv', index=False)
    summary.to_csv(out / f'e4_{heatmap_metric}_summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    heatmap(summary, out, heatmap_metric, args.center)
    lineplot(summary, out, heatmap_metric)
    if args.auc_gain_vs_random:
        auc_df = load(args.base_dir, 'auc', args.allow_partial)
        violin_auc_gain_vs_random(auc_df, out, args.violin_prototypes, args.violin_seed, args.gain_mode)
    else:
        violin(df, out, args.metric, args.violin_prototypes, args.violin_seed)
    if not args.no_statistics:
        write_statistics(args.base_dir, out, args.allow_partial, args.main_prototypes)
    print(f'\nSaved figures and tables to: {out}')

if __name__ == '__main__':
    main()

