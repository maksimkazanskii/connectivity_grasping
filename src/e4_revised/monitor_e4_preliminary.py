#!/usr/bin/env python3
"""Print provisional E4 results from in-progress per-seed CSVs."""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--results-dir',type=Path,required=True)
p.add_argument('--interval',type=int,default=60)
p.add_argument('--steps',type=int,default=100)
p.add_argument('--targets',type=int,default=500)
a=p.parse_args()

while True:
    try:
        rows=[]
        paths=sorted(a.results_dir.glob('seed_*/*.csv'))
        paths=[x for x in paths if not x.stem.endswith('_timing')]
        for path in paths:
            try:
                df=pd.read_csv(path,on_bad_lines='skip')
                if not {'seed','target','method','step','success','cumulative_success'}.issubset(df.columns):continue
                df['step']=pd.to_numeric(df.step,errors='coerce')
                df['success']=pd.to_numeric(df.success,errors='coerce')
                df['cumulative_success']=pd.to_numeric(df.cumulative_success,errors='coerce')
                for (seed,target,method),g in df.groupby(['seed','target','method']):
                    g=g.sort_values('step')
                    if len(g)!=a.steps or not np.array_equal(g.step.to_numpy(),np.arange(1,a.steps+1)):continue
                    if not np.array_equal(g.success.cumsum().to_numpy(),g.cumulative_success.to_numpy()):continue
                    rows.append(dict(scale=path.stem,seed=seed,target=str(target),method=method,
                                     successes=g.success.sum(),auc=g.cumulative_success.mean()))
            except (OSError,ValueError,pd.errors.ParserError):continue
        print('\n'+'='*90,flush=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'PRELIMINARY E4 results (only completed episodes)',flush=True)
        if not rows:
            print('No completed target episodes found yet.',flush=True)
        else:
            df=pd.DataFrame(rows)
            seed=df.groupby(['scale','method','seed'],as_index=False).agg(n_targets=('target','nunique'),successes=('successes','mean'),auc=('auc','mean'))
            summary=seed.groupby(['scale','method'],as_index=False).agg(seeds=('seed','nunique'),targets=('n_targets','sum'),successes=('successes','mean'),sd=('successes','std'),auc=('auc','mean'))
            summary['progress']=summary.targets.astype(str)+'/'+(summary.seeds*a.targets).astype(str)
            print(summary[['scale','method','seeds','progress','successes','sd','auc']].round(3).to_string(index=False),flush=True)
            print('SD across available seed means; preliminary, possibly unequal target counts.',flush=True)
        time.sleep(a.interval)
    except KeyboardInterrupt:
        print('\nStopped monitoring.')
        break
