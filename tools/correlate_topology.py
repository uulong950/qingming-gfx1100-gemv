#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

def num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def integer(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def read_distance(path):
    by = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                d = int(r["distance"])
                eta = float(r["eta_over_u"])
                max_abs = float(r["max_abs"])
                b = r["backend"]
            except (KeyError, ValueError):
                continue
            by[b].append({
                "distance": d,
                "eta": eta,
                "max_abs": max_abs,
                "exact_zero": integer(r.get("exact_zero", int(max_abs == 0.0))),
                "transition": integer(r.get("zero_to_nonzero", 0)),
                "phase": r.get("phase", ""),
            })
    for b in by:
        by[b].sort(key=lambda x: x["distance"])
    return by

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-csv", required=True)
    ap.add_argument("--isa-csv", required=True)
    ap.add_argument("--trace-csv", default="")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    data = read_distance(args.distance_csv)

    isa_rows = []
    try:
        with open(args.isa_csv, newline="") as f:
            isa_rows = list(csv.DictReader(f))
    except FileNotFoundError:
        pass

    trace_rows = []
    if args.trace_csv:
        try:
            with open(args.trace_csv, newline="") as f:
                trace_rows = list(csv.DictReader(f))
        except FileNotFoundError:
            pass

    out_rows = []
    for backend, vals in data.items():
        prev = None
        for cur in vals:
            if prev is not None and prev["eta"] > 0.0:
                ratio = cur["eta"] / prev["eta"]
                if ratio >= 2.0:
                    out_rows.append({
                        "backend": backend,
                        "distance": cur["distance"],
                        "phase": cur["phase"],
                        "eta_before": prev["eta"],
                        "eta_after": cur["eta"],
                        "ratio": ratio,
                        "distance_over_wave32": cur["distance"] / 32.0,
                        "zero_to_nonzero": cur["transition"],
                    })
            elif prev is not None and prev["exact_zero"] and not cur["exact_zero"]:
                out_rows.append({
                    "backend": backend,
                    "distance": cur["distance"],
                    "phase": cur["phase"],
                    "eta_before": 0.0,
                    "eta_after": cur["eta"],
                    "ratio": float("inf"),
                    "distance_over_wave32": cur["distance"] / 32.0,
                    "zero_to_nonzero": 1,
                })
            prev = cur

    csv_path = args.out_prefix + ".csv"
    md_path = args.out_prefix + ".md"
    fields = ["backend","distance","phase","eta_before","eta_after",
              "ratio","distance_over_wave32","zero_to_nonzero"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    matched = [r for r in isa_rows if integer(r.get("match_score", 0)) > 0]
    best = None
    if matched:
        best = max(
            matched,
            key=lambda r: (
                integer(r.get("trace_most_frequent", 0)),
                integer(r.get("match_score", 0)),
                integer(r.get("trace_dispatch_count", 0)),
            )
        )

    with open(md_path, "w") as f:
        f.write("# Cancellation-distance / ISA topology correlation\n\n")
        f.write("This report separates observed numerical transitions from ISA evidence. "
                "A numerical breakpoint is evidence about reduction ordering; it is not, "
                "by itself, proof of a specific internal tree.\n\n")

        f.write("## Numerical transitions\n\n")
        exact_events = []
        for backend, vals in data.items():
            for r in vals:
                if r["transition"]:
                    exact_events.append((backend, r))
        if exact_events:
            f.write("| backend | first nonzero distance | phase | max_abs | eta/u |\n")
            f.write("|---|---:|---|---:|---:|\n")
            for backend, r in exact_events:
                f.write(f"| {backend} | {r['distance']} | {r['phase']} | "
                        f"{r['max_abs']:.6g} | {r['eta']:.6g} |\n")
        else:
            f.write("No exact zero-to-nonzero transition marker was present in this sweep.\n")

        f.write("\n## >=2x eta/u steps\n\n")
        if out_rows:
            f.write("| backend | distance | phase | eta/u before | eta/u after | ratio | d/32 |\n")
            f.write("|---|---:|---|---:|---:|---:|---:|\n")
            for r in out_rows:
                ratio = "inf" if r["ratio"] == float("inf") else f"{r['ratio']:.3f}x"
                f.write(f"| {r['backend']} | {r['distance']} | {r['phase']} | "
                        f"{r['eta_before']:.6g} | {r['eta_after']:.6g} | "
                        f"{ratio} | {r['distance_over_wave32']:.3f} |\n")
        else:
            f.write("No >=2x eta/u step detected.\n")

        f.write("\n## Trace / ISA evidence\n\n")
        if trace_rows:
            f.write("| kernel | dispatches | LDS bytes | VGPR | AGPR | SGPR | WG | grid |\n")
            f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
            for r in trace_rows:
                wg = "x".join(str(r.get(k, "")) for k in
                             ("Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z"))
                grid = "x".join(str(r.get(k, "")) for k in
                               ("Grid_Size_X","Grid_Size_Y","Grid_Size_Z"))
                f.write(f"| `{r.get('kernel_name','')}` | {r.get('dispatch_count','')} | "
                        f"{r.get('LDS_Block_Size','')} | {r.get('VGPR_Count','')} | "
                        f"{r.get('Accum_VGPR_Count','')} | {r.get('SGPR_Count','')} | "
                        f"{wg} | {grid} |\n")
        else:
            f.write("No parsed kernel-trace summary was supplied.\n")

        f.write("\n### Disassembly matching\n\n")
        if best:
            f.write(f"- best traced kernel: `{best.get('matched_kernel','')}`\n")
            f.write(f"- matched code symbol: `{best.get('matched_symbol','')}`\n")
            f.write(f"- match score: **{best.get('match_score','')}**\n")
            f.write(f"- target/mode-frequency flag: **{best.get('trace_most_frequent','')}**\n")
            f.write(f"- LDS `ds_*` count: **{best.get('ds_ops','0')}**\n")
            f.write(f"- barriers: **{best.get('barrier','0')}**\n")
            f.write(f"- DPP indicators: **{best.get('dpp','0')}**\n")
            f.write(f"- permutation indicators: **{best.get('permute','0')}**\n")
            f.write(f"- global/flat atomics: **{best.get('global_atomic','0')}**\n\n")
            f.write("This is sufficient to correlate a traced kernel with a code-object "
                    "symbol. A precise reduction-tree claim should additionally map the "
                    "relevant instruction regions/stages, not just aggregate counts.\n")
        else:
            f.write("No trace-to-code-symbol match was established. Do not make an "
                    "ISA-specific reduction-topology claim from this capture alone.\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

if __name__ == "__main__":
    main()
