#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict

if len(sys.argv) != 2:
    print('usage: analyze.py results.csv')
    raise SystemExit(2)

path = sys.argv[1]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))

by_shape = defaultdict(dict)
for row in rows:
    by_shape[(int(row['M']), int(row['N']))][row['implementation']] = row

def number(row, key):
    try:
        return float(row[key])
    except Exception:
        return math.nan

print(f'file: {path}')
for impl in ('native', 'rocblas_sgemv', 'hipblaslt_gemm_as_gemv'):
    counts = Counter(r['status'] for r in rows if r['implementation'] == impl)
    if counts:
        print(f"{impl}: " + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))

print('\n=== native vs rocBLAS ===')
ratios = []
paths = defaultdict(list)
non_wins = []
for key, group in by_shape.items():
    native = group.get('native')
    rocblas = group.get('rocblas_sgemv')
    if not native or not rocblas:
        continue
    if native.get('status') != 'PASS' or rocblas.get('status') != 'PASS':
        continue
    nt = number(native, 'median_us')
    rt = number(rocblas, 'median_us')
    if nt <= 0 or rt <= 0:
        continue
    ratio = rt / nt
    ratios.append(ratio)
    paths[native.get('native_variant', 'NA')].append(ratio)
    if ratio <= 1.0:
        non_wins.append((ratio, key, nt, rt, native.get('native_variant', 'NA')))

if ratios:
    print(f'comparable PASS shapes={len(ratios)}')
    print(f'strict wins={sum(v > 1.0 for v in ratios)}')
    print(f'median={statistics.median(ratios):.4f}x')
    print(f'minimum={min(ratios):.4f}x')
    print(f'>0.5% wins={sum(v > 1.005 for v in ratios)}')
    print(f'within ±0.5%={sum(0.995 <= v <= 1.005 for v in ratios)}')
    print('\npath\tshapes\tmin\tmedian')
    for path_name, values in sorted(paths.items()):
        print(f'{path_name}\t{len(values)}\t{min(values):.4f}x\t{statistics.median(values):.4f}x')

if non_wins:
    print('\nnon-wins:')
    for ratio, (m, n), nt, rt, path_name in sorted(non_wins):
        print(f'  {m:8d} x {n:6d} {path_name:20s} native={nt:11.3f} us '
              f'rocBLAS={rt:11.3f} us ratio={ratio:.4f}x')

print('\n=== native correctness failures ===')
fails = [r for r in rows if r['implementation'] == 'native' and r['status'] == 'CORRECTNESS_FAIL']
if not fails:
    print('none')
else:
    for row in fails:
        print(f"  {row['M']} x {row['N']} path={row.get('native_variant','NA')} "
              f"failure_count={row.get('failure_count','NA')} max_abs={row.get('max_abs','NA')}")

print('\n=== native vs hipBLASLt on mutual PASS ===')
lt_ratios = []
for group in by_shape.values():
    native = group.get('native')
    lt = group.get('hipblaslt_gemm_as_gemv')
    if not native or not lt:
        continue
    if native.get('status') != 'PASS' or lt.get('status') != 'PASS':
        continue
    nt = number(native, 'median_us')
    lt_t = number(lt, 'median_us')
    if nt > 0 and lt_t > 0:
        lt_ratios.append(lt_t / nt)
if lt_ratios:
    print(f'mutual PASS={len(lt_ratios)} native wins={sum(v > 1.0 for v in lt_ratios)} '
          f'median={statistics.median(lt_ratios):.4f}x')


print('\n=== timing stability diagnostics ===')
unstable = []
for key, group in by_shape.items():
    for impl_name in ('native', 'rocblas_sgemv', 'hipblaslt_gemm_as_gemv'):
        row = group.get(impl_name)
        if not row or row.get('status') not in ('PASS', 'CORRECTNESS_FAIL'):
            continue
        med = number(row, 'median_us')
        mn = number(row, 'min_us')
        p95 = number(row, 'p95_us')
        sd = number(row, 'stddev_us')
        if not (math.isfinite(med) and med > 0):
            continue
        cv = sd / med if math.isfinite(sd) else math.nan
        spread = p95 / mn if math.isfinite(p95) and math.isfinite(mn) and mn > 0 else math.nan
        # Conservative flag: large coefficient of variation or broad
        # min-to-p95 span indicates a state/cache/clock mixture worth rerunning.
        if (math.isfinite(cv) and cv > 0.15) or (math.isfinite(spread) and spread > 1.50):
            unstable.append((key, impl_name, med, mn, p95, cv, spread))

if not unstable:
    print('none')
else:
    print(f'flagged rows={len(unstable)}')
    for (m, n), impl_name, med, mn, p95, cv, spread in sorted(unstable):
        print(f'  {m:8d} x {n:6d} {impl_name:24s} '
              f'median={med:10.3f} min={mn:10.3f} p95={p95:10.3f} '
              f'cv={cv:6.3f} p95/min={spread:6.3f}')
