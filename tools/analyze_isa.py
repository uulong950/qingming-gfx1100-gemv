#!/usr/bin/env python3
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

def safe_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return 0

def field(row, *names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return ""

def canonical(name):
    s = (name or "").strip().strip('"')
    if s.startswith("void "):
        s = s[5:].strip()
    # Demangled names: keep only the function spelling.
    if "(" in s and not s.startswith("_Z"):
        s = s.split("(", 1)[0].strip()
    # rocprofv3 --mangled-kernels commonly appends .kd.
    for suffix in (".kd", ".kd.kd"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    s = s.split("@", 1)[0]
    return s

def mangled_token(name):
    s = canonical(name)
    m = re.search(r"(_Z[A-Za-z0-9_.$]+)", s)
    return m.group(1) if m else ""

def symbol_score(trace_name, symbol):
    a = canonical(trace_name)
    b = canonical(symbol)
    if not a or not b:
        return 0
    if a == b:
        return 100
    ma, mb = mangled_token(a), mangled_token(b)
    if ma and mb and ma == mb:
        return 98
    if a in b or b in a:
        return 85
    if ma and ma in b:
        return 82
    if mb and mb in a:
        return 82
    return 0

def parse_symbols(isa_path):
    symbols = []
    text = isa_path.read_text(errors="replace")
    symbols.extend(re.findall(r"^[0-9a-fA-F]+ <([^>]+)>:", text, flags=re.MULTILINE))

    sym_path = isa_path.with_name(isa_path.name.replace(".isa.txt", ".symbols.txt"))
    if sym_path.exists():
        for line in sym_path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                candidate = parts[-1]
                if candidate not in ("U", "T", "t", "D", "d", "B", "b"):
                    symbols.append(candidate)

    # Stable de-dup preserving order.
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return text, out

def count(text, pattern):
    return len(re.findall(pattern, text, flags=re.MULTILINE))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-out", required=True)
    args = ap.parse_args()

    root = Path(args.capture_dir)
    traces = sorted(root.rglob("*kernel_trace.csv"))
    isa_files = sorted((root / "isa").glob("*.isa.txt"))

    dispatches = []
    for p in traces:
        try:
            with p.open(newline="", errors="replace") as f:
                for r in csv.DictReader(f):
                    name = field(r, "Kernel_Name", "Kernel Name", "Name", "kernel_name")
                    if name:
                        r["_kernel_name"] = name
                        r["_source"] = str(p)
                        dispatches.append(r)
        except Exception as e:
            print(f"warning: failed to parse {p}: {e}")

    counts = Counter(r["_kernel_name"] for r in dispatches)
    target_name = counts.most_common(1)[0][0] if counts else ""

    # Aggregate trace/resource facts by kernel name.
    by_name = defaultdict(list)
    for r in dispatches:
        by_name[r["_kernel_name"]].append(r)

    trace_fields = [
        "kernel_name", "canonical_name", "dispatch_count", "most_frequent",
        "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count",
        "SGPR_Count", "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
        "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"
    ]
    with open(args.trace_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trace_fields)
        w.writeheader()
        for name, rows in sorted(by_name.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            r0 = rows[0]
            w.writerow({
                "kernel_name": name,
                "canonical_name": canonical(name),
                "dispatch_count": len(rows),
                "most_frequent": int(name == target_name),
                "LDS_Block_Size": field(r0, "LDS_Block_Size"),
                "Scratch_Size": field(r0, "Scratch_Size"),
                "VGPR_Count": field(r0, "VGPR_Count"),
                "Accum_VGPR_Count": field(r0, "Accum_VGPR_Count"),
                "SGPR_Count": field(r0, "SGPR_Count"),
                "Workgroup_Size_X": field(r0, "Workgroup_Size_X"),
                "Workgroup_Size_Y": field(r0, "Workgroup_Size_Y"),
                "Workgroup_Size_Z": field(r0, "Workgroup_Size_Z"),
                "Grid_Size_X": field(r0, "Grid_Size_X"),
                "Grid_Size_Y": field(r0, "Grid_Size_Y"),
                "Grid_Size_Z": field(r0, "Grid_Size_Z"),
            })

    rows_out = []
    for isa in isa_files:
        text, symbols = parse_symbols(isa)
        best_trace = ""
        best_symbol = ""
        best_score = 0
        best_dispatch_count = 0

        for trace_name, trace_count in counts.items():
            for sym in symbols:
                score = symbol_score(trace_name, sym)
                if score > best_score or (score == best_score and trace_count > best_dispatch_count):
                    best_score = score
                    best_trace = trace_name
                    best_symbol = sym
                    best_dispatch_count = trace_count

        trace_rows = by_name.get(best_trace, [])
        tr = trace_rows[0] if trace_rows else {}

        rows_out.append({
            "isa_file": str(isa),
            "matched_kernel": best_trace if best_score else "",
            "matched_symbol": best_symbol if best_score else "",
            "match_score": best_score,
            "trace_dispatch_count": best_dispatch_count if best_score else 0,
            "trace_most_frequent": int(bool(best_score) and best_trace == target_name),
            "trace_LDS_Block_Size": field(tr, "LDS_Block_Size"),
            "trace_VGPR_Count": field(tr, "VGPR_Count"),
            "trace_Accum_VGPR_Count": field(tr, "Accum_VGPR_Count"),
            "trace_SGPR_Count": field(tr, "SGPR_Count"),
            "trace_Workgroup_Size_X": field(tr, "Workgroup_Size_X"),
            "trace_Workgroup_Size_Y": field(tr, "Workgroup_Size_Y"),
            "trace_Workgroup_Size_Z": field(tr, "Workgroup_Size_Z"),
            "trace_Grid_Size_X": field(tr, "Grid_Size_X"),
            "trace_Grid_Size_Y": field(tr, "Grid_Size_Y"),
            "trace_Grid_Size_Z": field(tr, "Grid_Size_Z"),
            "symbol_count": len(symbols),
            "v_fma_f32": count(text, r"\bv_fma_f32\b|\bv_fmac_f32\b"),
            "v_add_f32": count(text, r"\bv_add_f32\b"),
            "ds_ops": count(text, r"\bds_[A-Za-z0-9_]+\b"),
            "ds_read": count(text, r"\bds_read[A-Za-z0-9_]*\b"),
            "ds_write": count(text, r"\bds_write[A-Za-z0-9_]*\b"),
            "barrier": count(text, r"\bs_barrier\b"),
            "dpp": count(text, r"\bdpp\b|row_shr|row_shl|quad_perm|bank_mask|bound_ctrl"),
            "permute": count(text, r"\bv_permlane[A-Za-z0-9_]*\b|\bv_perm_b32\b|\bds_bpermute\b|\bds_permute\b"),
            "global_atomic": count(text, r"\bglobal_atomic_[A-Za-z0-9_]+\b|\bflat_atomic_[A-Za-z0-9_]+\b"),
            "global_load": count(text, r"\bglobal_load_[A-Za-z0-9_]+\b|\bflat_load_[A-Za-z0-9_]+\b"),
            "global_store": count(text, r"\bglobal_store_[A-Za-z0-9_]+\b|\bflat_store_[A-Za-z0-9_]+\b"),
        })

    if not rows_out:
        rows_out.append({
            "isa_file": "", "matched_kernel": "", "matched_symbol": "",
            "match_score": 0, "trace_dispatch_count": 0, "trace_most_frequent": 0,
            "trace_LDS_Block_Size": "", "trace_VGPR_Count": "",
            "trace_Accum_VGPR_Count": "", "trace_SGPR_Count": "",
            "trace_Workgroup_Size_X": "", "trace_Workgroup_Size_Y": "",
            "trace_Workgroup_Size_Z": "", "trace_Grid_Size_X": "",
            "trace_Grid_Size_Y": "", "trace_Grid_Size_Z": "",
            "symbol_count": 0, "v_fma_f32": 0, "v_add_f32": 0,
            "ds_ops": 0, "ds_read": 0, "ds_write": 0, "barrier": 0,
            "dpp": 0, "permute": 0, "global_atomic": 0,
            "global_load": 0, "global_store": 0,
        })

    fields = list(rows_out[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    matched = [r for r in rows_out if r["match_score"]]
    print(f"Kernel trace rows: {len(dispatches)}")
    print(f"Unique traced kernels: {len(counts)}")
    if target_name:
        print(f"Most frequent traced kernel: {target_name} ({counts[target_name]} dispatches)")
    print(f"Disassembly/code-symbol matches: {len(matched)}")
    if matched:
        best = max(matched, key=lambda r: (r["trace_most_frequent"], r["match_score"], r["trace_dispatch_count"]))
        print("Best match:")
        print(f"  trace : {best['matched_kernel']}")
        print(f"  symbol: {best['matched_symbol']}")
        print(f"  score : {best['match_score']}")
    else:
        print("No mangled/disassembly symbol match established.")
    print(f"Wrote {args.trace_out}")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
