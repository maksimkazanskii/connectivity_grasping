#!/usr/bin/env python3
"""E4 reference-memory scaling timing audit (no reruns).

Uses completed scale_N.csv trajectories and optional scale_N_timing.csv files.
Online time is end-to-end exploration time, NOT retrieval-only latency.
Training/memory costs are unavailable for legacy runs without timing CSVs.
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METHODS = ('connectivity_encoder','dynamic_fps_geometry_encoder',
           'dynamic_geometry_encoder','dynamic_random_geometry_encoder',
           'target_ca','random')
LABELS = ('Connectivity','Dynamic FPS','Dynamic geometry','Dynamic random',
          'Target CA','Random')
SCALE = re.compile(r'scale_(\d+)\.csv$')

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) and v>=0 else None
    except (TypeError,ValueError):
        return None

def fmt(values, unit=1, digits=2):
    if not values:return '--'
    m=mean(values)*unit
    if len(values)==1:return f'{m:.{digits}f}'
    return f'{m:.{digits}f} ± {stdev(values)*unit:.{digits}f}'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-dir',type=Path,default=Path('results/e4_reference_scaling_16'))
    p.add_argument('--out-dir',type=Path,default=Path('results/e4_timing_reference_analysis'))
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--expected-targets',type=int,default=500)
    p.add_argument('--prototypes-per-object',type=int,default=16)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted(a.reference_dir.glob('seed_*/scale_*.csv'))
    paths=[x for x in paths if SCALE.fullmatch(x.name)]
    if not paths:raise SystemExit(f'No seed_*/scale_N.csv files under {a.reference_dir}')
    per_seed=[]; audit=[]
    for path in paths:
        n=int(SCALE.fullmatch(path.name).group(1))
        seed_match=re.fullmatch(r'seed_(\d+)',path.parent.name)
        if not seed_match:continue
        seed=int(seed_match.group(1))
        rows=read(path)
        groups=defaultdict(list)
        for r in rows:
            if r.get('method') in METHODS:
                groups[(r['target'],r['method'])].append(r)
        timing_path=path.with_name(path.stem+'_timing.csv')
        timing={}
        if timing_path.exists():
            for r in read(timing_path):
                timing[(r.get('target'),r.get('method'))]=r
        log=path.with_suffix('.log')
        actual=None
        if log.exists():
            m=re.search(r'memory prototypes=(\d+)',log.read_text(errors='replace'))
            if m:actual=int(m.group(1))
        for method in METHODS:
            times=[]; trains=[]; builds=[]; actuals=[]; valid=0
            for (target,met),g in groups.items():
                if met!=method:continue
                try:g.sort(key=lambda r:int(r['step']))
                except (ValueError,KeyError):continue
                if len(g)!=a.steps or [int(r['step']) for r in g]!=list(range(1,a.steps+1)):
                    continue
                valid+=1
                t=timing.get((target,method),{})
                episode=number(t.get('inference_episode_sec'))
                if episode is None:episode=number(g[-1].get('runtime_sec'))
                if episode is not None:times.append(1000*episode/a.steps)
                for field,values in [('training_sec',trains),('memory_build_sec',builds)]:
                    v=number(t.get(field))
                    if v is not None:values.append(v)
                av=number(t.get('actual_prototypes'))
                if av is not None:actuals.append(av)
            # Training and memory build are RUN-LEVEL costs repeated across targets;
            # count only one per seed and method, never sum over target episodes.
            complete=valid==a.expected_targets
            if not complete:audit.append(f'INCOMPLETE {path} {method}: {valid}/{a.expected_targets}')
            per_seed.append(dict(reference_objects=n,nominal_prototypes=n*a.prototypes_per_object,
                actual_prototypes=actual if actual is not None else (mean(actuals) if actuals else ''),
                seed=seed,method=method,targets=valid,complete=complete,
                online_ms_per_step=mean(times) if times and complete else '',
                training_sec=trains[0] if trains and complete else '',
                memory_build_sec=builds[0] if builds and complete else '',
                timing_source='timing_csv' if timing else 'runtime_sec'))
    with (a.out_dir/'e4_timing_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
    scales=sorted({r['reference_objects'] for r in per_seed})
    def values(n,m,field):
        return [float(r[field]) for r in per_seed if r['reference_objects']==n and r['method']==m and r['complete'] and r[field]!='']
    lines=[]
    for metric,title,units in [('online_ms_per_step','End-to-end online evaluation time','ms/step'),
                               ('training_sec','Offline model training time','s'),
                               ('memory_build_sec','Reference memory construction time','s')]:
        lines.extend([r'\begin{table*}[t]',r'\centering',r'\caption{'+title+r' versus reference-memory size ('+units+r'; mean $\pm$ SD across independent seeds).}',
                      r'\label{tab:e4_memory_'+metric+r'}',r'{\scriptsize',r'\setlength{\tabcolsep}{5pt}',
                      r'\renewcommand{\arraystretch}{1.15}',r'\begin{tabular}{r r '+ 'c'*len(METHODS)+'}',r'\hline',
                      'References & Nominal prototypes & '+' & '.join(LABELS)+r' \\',r'\hline'])
        for n in scales:
            lines.append(f'{n} & {n*a.prototypes_per_object:,} & '+' & '.join(fmt(values(n,m,metric)) for m in METHODS)+r' \\')
        lines.extend([r'\hline',r'\end{tabular}',r'}',r'\end{table*}',''])
    (a.out_dir/'e4_timing_tables.tex').write_text('\n'.join(lines))
    audit.append(f'Analyzed {len(paths)} scale CSVs, {len(per_seed)} method/seed/scale records')
    audit.append('Online timing = full exploration episode elapsed / completed steps; NOT retrieval-only latency.')
    audit.append('Training and memory-build values require scale_N_timing.csv; missing values remain --.')
    audit.append('Nominal prototypes = reference objects × per-object cap; actual may be smaller.')
    (a.out_dir/'e4_timing_audit.txt').write_text('\n'.join(audit)+'\n')
    print('\n'.join(audit[-4:]))
    print('Wrote',a.out_dir/'e4_timing_per_seed.csv')
    print('Wrote',a.out_dir/'e4_timing_tables.tex')

if __name__=='__main__':main()
