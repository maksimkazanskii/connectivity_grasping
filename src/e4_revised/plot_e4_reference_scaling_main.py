#!/usr/bin/env python3
"""Reference-object scaling: one Success@100 line plot with seed SD, and stacked Success/AUC LaTeX table.

Self-contained; requires numpy, pandas, matplotlib, scipy.
Uses only fully completed target trajectories and seed runs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

FILE_RE = re.compile(r"^scale_(\d+)\.csv$")
SEED_RE = re.compile(r"^seed_(\d+)$")
LOG_RE = re.compile(r"memory prototypes=(\d+) regions=(\d+)")

def load_episodes(folder, steps, expected_targets, training_objects, memory_budget):
    paths = sorted(folder.glob('seed_*/scale_*.csv')) + sorted(folder.glob('scale_*.csv'))
    paths = list(dict.fromkeys(paths))
    rows = []
    configs = set()
    incomplete = 0
    for path in paths:
        match = FILE_RE.fullmatch(path.name)
        if not match:
            continue
        scale = int(match.group(1))
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f'WARNING: cannot read {path}: {exc}')
            continue
        required = {'seed', 'target', 'method', 'step', 'success', 'cumulative_success'}
        if not required.issubset(df.columns):
            print(f'WARNING: {path} missing {sorted(required - set(df.columns))}')
            continue
        df = df[df.method.isin(METHODS)].copy()
        if df.empty:
            continue
        for col in ('step', 'success', 'cumulative_success'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        seed_match = SEED_RE.fullmatch(path.parent.name)
        if seed_match and any(str(s) != seed_match.group(1) for s in df.seed.unique()):
            print(f'WARNING: seed mismatch in {path}; skipping')
            continue
        log = path.with_suffix('.log')
        text = log.read_text(errors='replace') if log.exists() else ''
        m = LOG_RE.search(text)
        actual = float(m.group(1)) if m else np.nan
        regions = float(m.group(2)) if m else np.nan
        for (seed, target, method), group in df.groupby(['seed', 'target', 'method'], sort=False):
            group = group.sort_values('step')
            if len(group) != steps or not np.array_equal(group.step.to_numpy(), np.arange(1, steps + 1)):
                incomplete += 1
                continue
            if not np.allclose(group.success.cumsum(), group.cumulative_success, equal_nan=False):
                print(f'WARNING: inconsistent cumulative success: {path} seed={seed} target={target} method={method}')
                continue
            runtime = pd.to_numeric(group['runtime_sec'], errors='coerce') if 'runtime_sec' in group else pd.Series(dtype=float)
            rows.append(dict(reference_objects=scale, training_objects=(scale if training_objects == 0 else training_objects),
                             memory_budget=scale * memory_budget, prototypes_per_object=memory_budget, seed=str(seed), target=str(target),
                             method=method, successes=float(group.success.sum()),
                             auc=float(group.cumulative_success.mean()),
                             runtime_sec=float(runtime.iloc[-1]) if len(runtime) and pd.notna(runtime.iloc[-1]) else np.nan,
                             actual_prototypes=actual, connected_regions=regions,
                             source_file=str(path)))
        configs.add((scale, str(path)))
    print(f'Read {len(configs)} run CSVs; {len(rows)} complete method-target episodes; skipped {incomplete} incomplete episodes.')
    if not rows:
        raise SystemExit('No complete episodes. Check --results-dir, --steps and CSV format.')
    episodes = pd.DataFrame(rows)
    counts = episodes.groupby(['reference_objects', 'method', 'seed']).target.nunique()
    for (scale, method, seed), count in counts.items():
        if count != expected_targets:
            print(f'WARNING: reference={scale} method={method} seed={seed}: {count}/{expected_targets} targets; excluded from seed statistics')
    return episodes


def summarize(episodes, expected_targets):
    keys = ['reference_objects', 'method', 'seed']
    per_seed = episodes.groupby(keys, as_index=False).agg(
        successes=('successes', 'mean'), auc=('auc', 'mean'),
        runtime_sec=('runtime_sec', 'mean'), n_targets=('target', 'nunique'),
        actual_prototypes=('actual_prototypes', 'median'),
        connected_regions=('connected_regions', 'median'))
    per_seed['complete_seed'] = per_seed.n_targets.eq(expected_targets)
    valid = per_seed[per_seed.complete_seed]
    if valid.empty:
        raise SystemExit('No configurations have the required number of complete targets per seed.')
    summary = valid.groupby(['reference_objects', 'method'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), successes_mean=('successes', 'mean'),
        successes_sd=('successes', 'std'), auc_mean=('auc', 'mean'),
        auc_sd=('auc', 'std'), runtime_sec_mean=('runtime_sec', 'mean'),
        actual_prototypes=('actual_prototypes', 'median'),
        connected_regions=('connected_regions', 'median'))
    return per_seed, summary


METHODS = ['connectivity_encoder','dynamic_fps_geometry_encoder','dynamic_geometry_encoder',
           'dynamic_random_geometry_encoder','target_ca','random']
LABELS = {'connectivity_encoder':'Connectivity','dynamic_fps_geometry_encoder':'Dynamic FPS',
          'dynamic_geometry_encoder':'Dynamic geometry','dynamic_random_geometry_encoder':'Dynamic random geometry',
          'target_ca':'Target CA','random':'Random'}
# Exact palette from plot_e4_prototype_ablation(7).py
COLORS = {'connectivity_encoder':'#2868A5','dynamic_fps_geometry_encoder':'#72B4D6',
          'dynamic_geometry_encoder':'#648A92','dynamic_random_geometry_encoder':'#EF956B',
          'target_ca':'#B51F36','random':'#777777'}
MARKERS = ['o','s','^','D','v','P']

def holm(ps):
    ps=np.asarray(ps,float)
    order=np.argsort(ps)
    result=np.empty(len(ps))
    result[order]=np.maximum.accumulate(np.minimum(1,ps[order]*(len(ps)-np.arange(len(ps)))))
    return result

def calculate_stats(per_seed, scales, min_seeds):
    valid=per_seed[per_seed.complete_seed & per_seed.method.isin(METHODS)].copy()
    # Retain scales with all six methods and enough complete seeds.
    accepted=[]
    for scale in scales:
        x=valid[valid.reference_objects.eq(scale)]
        if all(x[x.method.eq(m)].seed.nunique()>=min_seeds for m in METHODS):
            accepted.append(scale)
        else:
            print(f'WARNING: skipping N={scale}: at least one method has fewer than {min_seeds} complete seeds')
    if not accepted: raise ValueError('No reference scales have all six methods with enough complete seeds')
    valid=valid[valid.reference_objects.isin(accepted)]
    sig=[]
    for metric in ['successes','auc']:
        for scale in accepted:
            wide=valid[valid.reference_objects.eq(scale)].pivot(index='seed',columns='method',values=metric)
            pvals=[]; baselines=[]
            for baseline in METHODS[1:]:
                diff=(wide['connectivity_encoder']-wide[baseline]).dropna()
                p=float(ttest_1samp(diff,0).pvalue) if len(diff)>=3 and diff.std(ddof=1)>0 else (0. if len(diff)>=3 and diff.mean()!=0 and diff.std(ddof=1)==0 else np.nan)
                baselines.append((baseline,len(diff),diff.mean(),p))
                pvals.append(p)
            adjusted=np.full(len(pvals),np.nan)
            inds=np.flatnonzero(np.isfinite(pvals))
            if len(inds): adjusted[inds]=holm(np.asarray(pvals)[inds])
            for (baseline,n,delta,p),adj in zip(baselines,adjusted):
                sig.append(dict(metric=metric,reference_objects=scale,baseline=baseline,n_paired_seeds=n,
                                delta=delta,p_raw=p,p_holm=adj))
    return valid,accepted,pd.DataFrame(sig)

def make_plot(summary, scales, out):
    fig,ax=plt.subplots(figsize=(9.2,5.3))
    for i,m in enumerate(METHODS):
        part=summary[(summary.method==m)&summary.reference_objects.isin(scales)].sort_values('reference_objects')
        x=part.reference_objects.to_numpy(float); y=part.successes_mean.to_numpy(float)
        sd=part.successes_sd.to_numpy(float)
        ax.plot(x,y,marker=MARKERS[i],lw=2.2,ms=6,color=COLORS[m],label=LABELS[m],zorder=3)
        ok=np.isfinite(sd)
        if ok.any():ax.fill_between(x[ok],y[ok]-sd[ok],y[ok]+sd[ok],color=COLORS[m],alpha=.17,lw=0,zorder=2)
    ax.set_xscale('log',base=2)
    ax.set_xticks(scales,labels=[str(x) for x in scales])
    ax.set_xlabel('Number of reference objects')
    ax.set_ylabel('Success@100 (%)')
    ax.grid(axis='y',ls='--',alpha=.25,lw=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,ncol=2,fontsize=8.7,loc='best')
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(out/f'e4_reference_scaling_success_lines_sd.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def latex_table(summary, scales, stats, min_seeds):
    """Match the prototype table: vertically stacked, centered section titles, horizontal rules."""
    def cell(metric, method, n):
        subset = summary[summary.reference_objects.eq(n)]
        row = subset[subset.method.eq(method)].iloc[0]
        mu, sd = float(row[f'{metric}_mean']), float(row[f'{metric}_sd'])
        value = f'{mu:.2f}\\pm{sd:.2f}' if np.isfinite(sd) else f'{mu:.2f}'
        best = mu >= subset[f'{metric}_mean'].max() - 1e-10
        if best:
            value = r'\mathbf{' + value + '}'
        if method != 'connectivity_encoder':
            p = stats[(stats.metric.eq(metric)) &
                      (stats.reference_objects.eq(n)) &
                      (stats.baseline.eq(method))]
            if not p.empty and np.isfinite(p.iloc[0].p_holm) and p.iloc[0].p_holm < .01:
                value += r'^{**}'
        return '$' + value + '$'

    lines = [
        r'% Requires \usepackage{booktabs}',
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{E4 reference-object scaling at $P=16$ prototypes per reference object. '
         r'Entries are mean $\pm$ SD across independently trained seeds '
         + f'(at least {min_seeds} complete seeds per included reference count). '
         + r'Bold indicates the highest mean within each column and metric. '
         + r'$^{**}$ denotes Holm-adjusted $p<0.01$ against connectivity in '
         + r'two-sided paired seed-level $t$-tests; correction is applied across '
         + r'five baselines separately for each metric and reference count.}'),
        r'\label{tab:e4_reference_scaling}',
        r'\scriptsize' if len(scales) > 6 else r'\small',
        r'\setlength{\tabcolsep}{3.8pt}' if len(scales) > 6 else r'\setlength{\tabcolsep}{9pt}',
        r'\renewcommand{\arraystretch}{1.18}',
        r'\begin{tabular}{l' + 'c'*len(scales) + '}',
        r'\toprule',
        ]
    for metric, title in [('successes', r'(a) Success@100 (\%)'),
                          ('auc', r'(b) Cumulative-success AUC')]:
        if metric == 'auc':
            lines.append(r'\midrule[1.1pt]')
        lines.append(r'\multicolumn{' + str(len(scales)+1) + r'}{c}{\textbf{' + title + r'}} \\')
        lines.append(r'\midrule')
        lines.append('Method & ' + ' & '.join(f'$N={n}$' for n in scales) + r' \\')
        lines.append(r'\midrule')
        for method in METHODS:
            label = r'\textbf{Connectivity}' if method == 'connectivity_encoder' else LABELS[method]
            lines.append(label + ' & ' + ' & '.join(cell(metric, method, n) for n in scales) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    return '\n'.join(lines) + '\n'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-dir',type=Path,default=Path('results/e4_reference_scaling_16'))
    ap.add_argument('--out',type=Path,default=Path('results/e4_reference_scaling_16/figures'))
    ap.add_argument('--targets',type=int,default=500)
    ap.add_argument('--min-seeds',type=int,default=5)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    episodes=load_episodes(args.results_dir,100,args.targets,0,16)
    per_seed,summary=summarize(episodes,args.targets)
    scales=sorted(summary.reference_objects.unique())
    valid,accepted,stats=calculate_stats(per_seed,scales,args.min_seeds)
    summary=summary[summary.reference_objects.isin(accepted)&summary.method.isin(METHODS)]
    # Use the exact accepted seeds to avoid plotting partial results.
    summary=valid.groupby(['reference_objects','method'],as_index=False).agg(
        n_seeds=('seed','nunique'),successes_mean=('successes','mean'),successes_sd=('successes','std'),
        auc_mean=('auc','mean'),auc_sd=('auc','std'))
    make_plot(summary,accepted,args.out)
    (args.out/'e4_reference_scaling_success_auc_stacked.tex').write_text(latex_table(summary,accepted,stats,args.min_seeds))
    summary.to_csv(args.out/'e4_reference_scaling_summary.csv',index=False)
    stats.to_csv(args.out/'e4_reference_scaling_paired_statistics.csv',index=False)
    print(f'Saved plot PDF/PNG, stacked LaTeX table, summary and p-values to {args.out}')
    print(f'Included reference counts: {accepted}')
if __name__=='__main__':main()
