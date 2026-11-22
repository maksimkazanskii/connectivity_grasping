#!/usr/bin/env python3
"""Analyze E4 joint scaling, paired outcomes, offline costs and online inference.

Each scale trains two models ONCE; seeds/targets are evaluation episodes, not
independent training repetitions. Timing CSVs are emitted by timing-enabled E4.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('random', 'target_ca', 'random_geometry_encoder', 'fps_geometry_encoder', 'geometry_encoder', 'dynamic_random_geometry_encoder', 'dynamic_fps_geometry_encoder', 'dynamic_geometry_encoder', 'connectivity_encoder')
LABELS = {'random':'Random','target_ca':'Target CA','random_geometry_encoder':'Random-prototype geometry','fps_geometry_encoder':'FPS geometry','geometry_encoder':'Geometry encoder','dynamic_random_geometry_encoder':'Dynamic random geometry','dynamic_fps_geometry_encoder':'Dynamic FPS geometry','dynamic_geometry_encoder':'Dynamic geometry','connectivity_encoder':'Connectivity encoder'}
RUN_RE = re.compile(r'^scale_(\d+)\.csv$')
OLD_MEMORY_RE = re.compile(r'^memory_(\d+)\.csv$')
OLD_OBJECT_RE = re.compile(r'^refs_(\d+)(?:_train_(\d+))?\.csv$')
LOG_RE = re.compile(r'memory prototypes=(\d+) regions=(\d+)')

def ci(x):
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]
    if len(a)<2:return (np.nan,np.nan)
    d=stats.t.ppf(.975,len(a)-1)*stats.sem(a)
    return (float(a.mean()-d),float(a.mean()+d))

def configuration(path,kind):
    m=RUN_RE.fullmatch(path.name)
    if m:
        s=int(m.group(1));return dict(scale=s,reference_objects=s,training_objects=s,memory_budget=s)
    m=(OLD_MEMORY_RE if kind=='memory' else OLD_OBJECT_RE).fullmatch(path.name)
    if not m:return None
    if kind=='memory':
        s=int(m.group(1));return dict(scale=s,reference_objects=400,training_objects=400,memory_budget=s)
    r=int(m.group(1)); t=int(m.group(2)) if m.group(2) else 300
    return dict(scale=r,reference_objects=r,training_objects=t,memory_budget=2000)

def load_one(path,kind):
    conf=configuration(path,kind)
    if conf is None:return [],[],None
    raw=pd.read_csv(path)
    needed={'seed','target','method','step','success','cumulative_success'}
    if not needed.issubset(raw.columns):raise ValueError(f'missing {sorted(needed-set(raw.columns))}')
    raw=raw[raw.method.isin(METHODS)].copy()
    if raw.empty:raise ValueError('no recognized methods')
    for c in ('step','success','cumulative_success'):
        raw[c]=pd.to_numeric(raw[c],errors='raise')
    if raw.duplicated(['seed','target','method','step']).any():raise ValueError('duplicate steps')
    log=path.with_suffix('.log'); proto=regions=np.nan
    if log.exists():
        m=LOG_RE.search(log.read_text(errors='replace'))
        if m:proto,regions=map(int,m.groups())
    episodes=[]
    for (seed,target,method),g in raw.groupby(['seed','target','method'],sort=False):
        g=g.sort_values('step'); n=len(g)
        if not np.array_equal(g.step.to_numpy(),np.arange(1,n+1)):raise ValueError('nonconsecutive steps')
        if not np.array_equal(g.success.cumsum().to_numpy(),g.cumulative_success.to_numpy()):raise ValueError('incorrect cumulative successes')
        row=dict(conf,seed=seed,target=str(target),method=method,steps=n,
                 successes=float(g.success.sum()),success_rate=float(g.success.mean()),
                 auc=float(g.cumulative_success.mean()),normalized_auc=float(g.cumulative_success.mean()/n),
                 actual_prototypes=proto,connected_regions=regions)
        if 'runtime_sec' in g:row['runtime_sec']=float(g.runtime_sec.iloc[-1])
        episodes.append(row)
    timing_path=path.with_name(path.stem+'_timing.csv')
    timing=[]
    if timing_path.exists():
        t=pd.read_csv(timing_path)
        req={'seed','target','method','training_sec','memory_build_sec','preparation_sec','inference_episode_sec','inference_step_ms'}
        if not req.issubset(t.columns):raise ValueError(f'{timing_path}: missing {sorted(req-set(t.columns))}')
        t=t[t.method.isin(METHODS)].copy()
        for r in t.to_dict('records'):
            r.update(conf);r['target']=str(r['target']);r['source_file']=str(timing_path)
            timing.append(r)
    return episodes,timing,conf

def aggregate(df,keys,metrics):
    rows=[]
    for key,g in df.groupby(keys,dropna=False):
        r=dict(zip(keys,key if isinstance(key,tuple) else (key,)));r['episodes']=len(g)
        for metric in metrics:
            if metric not in g:continue
            vals=pd.to_numeric(g[metric],errors='coerce').dropna()
            r[metric+'_mean']=vals.mean() if len(vals) else np.nan
            r[metric+'_sd']=vals.std(ddof=1) if len(vals)>1 else np.nan
            r[metric+'_ci95_low'],r[metric+'_ci95_high']=ci(vals)
        rows.append(r)
    return pd.DataFrame(rows)

def paired(df):
    rows=[]
    for scale,g in df.groupby('scale'):
        for baseline in ('random','target_ca','random_geometry_encoder','fps_geometry_encoder','geometry_encoder'):
            for metric in ('successes','auc','normalized_auc'):
                p=g.pivot_table(index=['seed','target'],columns='method',values=metric,aggfunc='first')
                if 'connectivity_encoder' not in p or baseline not in p:continue
                both=p[['connectivity_encoder',baseline]].dropna()
                if both.empty:continue
                diff=(both.iloc[:,0]-both.iloc[:,1]).to_numpy()
                lo,hi=ci(diff)
                pv=float(stats.ttest_1samp(diff,0).pvalue) if len(diff)>1 and diff.std(ddof=1)>0 else (1. if np.allclose(diff,0) else np.nan)
                rows.append(dict(scale=scale,baseline=baseline,metric=metric,n_pairs=len(diff),mean_difference=diff.mean(),ci95_low=lo,ci95_high=hi,p_value=pv,wins=int((diff>0).sum()),losses=int((diff<0).sum()),ties=int((diff==0).sum())))
    return pd.DataFrame(rows)

def plot_lines(summary,metric,path,ylabel=None):
    if metric+'_mean' not in summary:return
    fig,ax=plt.subplots(figsize=(8,5))
    for method in METHODS:
        g=summary[summary.method==method].sort_values('scale')
        if g.empty:continue
        ax.plot(g.scale,g[metric+'_mean'],marker='o',label=LABELS[method])
        if metric=='successes':
            lo=g[metric+'_ci95_low'].to_numpy(dtype=float);hi=g[metric+'_ci95_high'].to_numpy(dtype=float)
            ax.fill_between(g.scale.to_numpy(),lo,hi,alpha=.1)
    ax.set_xlabel('Joint scale (reference objects = training objects = prototype budget)')
    ax.set_ylabel(ylabel or metric.replace('_',' ').title())
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def process(folder,kind,out):
    if not folder.exists():print('WARNING: missing',folder);return
    paths=sorted(p for p in folder.glob('*.csv') if configuration(p,kind) is not None)
    episodes=[]; timings=[]; configs=[]
    for path in paths:
        try:
            e,t,c=load_one(path,kind);episodes.extend(e);timings.extend(t);configs.append(c)
        except Exception as ex:print(f'WARNING: {path}: {ex}')
    if not episodes:print('WARNING: no valid runs in',folder);return
    df=pd.DataFrame(episodes)
    df.to_csv(out/f'{kind}_per_object.csv',index=False)
    summ=aggregate(df,['scale','reference_objects','training_objects','memory_budget','method'],['successes','success_rate','auc','normalized_auc'])
    counts=df.groupby(['scale','method']).size().rename('objects').reset_index()
    summ=summ.merge(counts,on=['scale','method'])
    summ=summ.merge(df.groupby(['scale','method'])[['actual_prototypes','connected_regions']].first().reset_index(),on=['scale','method'])
    summ.to_csv(out/f'{kind}_summary.csv',index=False)
    paired(df).to_csv(out/f'{kind}_paired_tests.csv',index=False)
    for metric in ('successes','auc','normalized_auc'):
        plot_lines(summ,metric,out/f'{kind}_{metric}.png')
    print(f'\n{kind.upper()}: {len(configs)} configurations, {len(df)} method-target evaluations')
    print(summ[['scale','method','objects','successes_mean','auc_mean','actual_prototypes']].round(3).to_string(index=False))
    if not timings:
        print('WARNING: no *_timing.csv files; training/inference timing unavailable');return
    td=pd.DataFrame(timings)
    td.to_csv(out/f'{kind}_timing_per_episode.csv',index=False)
    tm=aggregate(td,['scale','method'],['inference_episode_sec','inference_step_ms'])
    # Offline costs are shared across episodes, not independent repetitions.
    offline=[]
    for (scale,method),g in td.groupby(['scale','method']):
        vals={}
        for field in ('training_sec','memory_build_sec','preparation_sec','actual_prototypes','connected_regions','reference_objects','training_objects','memory_budget'):
            if field in g:
                u=pd.to_numeric(g[field],errors='coerce').dropna().unique()
                vals[field]=float(u[0]) if len(u)==1 else np.nan
                if len(u)>1:print(f'WARNING: varying {field} for scale={scale}, method={method}')
        offline.append(dict(scale=scale,method=method,**vals))
    tm=tm.merge(pd.DataFrame(offline),on=['scale','method'])
    tm['total_offline_sec']=tm.preparation_sec+tm.training_sec
    tm.to_csv(out/f'{kind}_timing_summary.csv',index=False)
    for metric in ('inference_episode_sec','inference_step_ms'):
        plot_lines(tm,metric,out/f'{kind}_{metric}.png')
    training=tm[tm.method.isin(['random_geometry_encoder','fps_geometry_encoder','geometry_encoder','dynamic_random_geometry_encoder','dynamic_fps_geometry_encoder','dynamic_geometry_encoder','connectivity_encoder'])][['scale','method','training_sec','memory_build_sec','preparation_sec','total_offline_sec']].copy()
    training['training_replicates']=1
    training.to_csv(out/f'{kind}_training_runs.csv',index=False)
    print(f'Training sessions measured: {len(training)} (one per encoder per configuration)')
    print(tm[['scale','method','training_sec','inference_episode_sec_mean','inference_step_ms_mean']].round(4).to_string(index=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--objects-dir',type=Path,default=Path('results/e4_r_object_ablations'))
    ap.add_argument('--memory-dir',type=Path,default=Path('results/e4_memory_budget_ablations'))
    ap.add_argument('--out',type=Path,default=Path('results/e4_ablation_stats'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    process(args.objects_dir,'objects',args.out)
    process(args.memory_dir,'memory',args.out)
    print('\nSaved results to',args.out)
if __name__=='__main__':main()
