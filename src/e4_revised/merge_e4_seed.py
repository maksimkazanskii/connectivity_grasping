#!/usr/bin/env python3
"""Append one completed per-seed run into the scale-level CSVs and print seed means."""
import argparse, csv
from pathlib import Path
from collections import defaultdict

p=argparse.ArgumentParser()
p.add_argument('--folder',type=Path,required=True)
p.add_argument('--scale',type=int,required=True)
p.add_argument('--seed',type=int,required=True)
a=p.parse_args()
stem=f'scale_{a.scale}'
source=a.folder/f'seed_{a.seed}'/f'{stem}.csv'
if not source.exists():raise SystemExit(f'Missing {source}')
summary=defaultdict(list)
for suffix in ('', '_timing'):
    src=source.with_name(stem+suffix+'.csv')
    dst=a.folder/(stem+suffix+'.csv')
    with src.open(newline='') as fi:
        reader=csv.DictReader(fi)
        if dst.exists():
            with dst.open(newline='') as check:
                header=next(csv.reader(check))
            if header != reader.fieldnames:raise SystemExit(f'Header mismatch: {dst}')
        with dst.open('a',newline='') as fo:
            writer=csv.DictWriter(fo,fieldnames=reader.fieldnames)
            if fo.tell()==0:writer.writeheader()
            for row in reader:
                writer.writerow(row)
                if not suffix and int(row['step'])==100:
                    summary[row['method']].append(float(row['cumulative_success']))
print(f'\nCOMPLETED seed={a.seed}, scale={a.scale} — mean successes/100:')
for method,values in sorted(summary.items()):
    print(f'  {method:28s} {sum(values)/len(values):7.2f} (n={len(values)})')
