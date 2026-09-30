"""Measure warm local query latency over one natural-language query per line."""
from __future__ import annotations
import argparse
import platform
import statistics
import time
from pathlib import Path

from codelens import db_open,search


def percentile(values,p):
    values=sorted(values)
    if not values: return 0.0
    at=(len(values)-1)*p
    lo=int(at);hi=min(len(values)-1,lo+1);fraction=at-lo
    return values[lo]*(1-fraction)+values[hi]*fraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default='.codelens/index.sqlite')
    parser.add_argument('--revision')
    parser.add_argument('--queries',type=Path,required=True,help='UTF-8 text file, one query per line')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--top-k',type=int,default=10)
    args=parser.parse_args()
    queries=[line.strip() for line in args.queries.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not queries: parser.error('query file must contain at least one non-empty line')
    if args.repeat<1: parser.error('--repeat must be at least 1')
    with db_open(args.db) as db:
        block_count=db.execute('SELECT COUNT(*) FROM blocks'+(' WHERE revision=?' if args.revision else ''),([args.revision] if args.revision else [])).fetchone()[0]
    if not block_count: parser.error('no indexed blocks found for the selected database and revision')
    # One warm-up initializes the model and filesystem/SQLite caches before timing.
    search(queries[0],args.revision,args.top_k,args.db)
    timings=[]
    for _ in range(args.repeat):
        for query in queries:
            started=time.perf_counter();search(query,args.revision,args.top_k,args.db)
            timings.append((time.perf_counter()-started)*1000)
    print(f'Queries: {len(queries)} | repetitions: {args.repeat} | indexed blocks: {block_count}')
    print(f'Warm query latency: p50={statistics.median(timings):.1f} ms | p95={percentile(timings,.95):.1f} ms')
    print(f'Platform: {platform.platform()} | Python: {platform.python_version()} | CPU-only embedding model')
    print('Warm latency excludes initial model download/load and index construction.')


if __name__=='__main__': main()
