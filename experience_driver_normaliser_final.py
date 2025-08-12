#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ED Normaliser — Day1/DayN + God-tier Compression
- Theme-scoped
- Component-aware (Category & Subcategory)
- Zero curated lists (no contrastives/qualifiers/synonyms)
- Deterministic merges via symmetric top-1 NN + thresholds
"""

import os, re, json, argparse
from collections import defaultdict
from datetime import datetime
import pandas as pd

# ---------- allowed hard-coded numbers (CLI-overridable) ----------
THRESH_CAT = 0.90
THRESH_SUB = 0.90

# ---------- similarity (RapidFuzz -> difflib fallback) ----------
try:
    from rapidfuzz import fuzz
    def _sim(a: str, b: str) -> float:
        return fuzz.token_sort_ratio(a, b) / 100.0
except Exception:
    import difflib
    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- neutral text utils ----------
DELIM_REGEX = r"\s*(?:→|->|>|/|:)\s*"

def _norm(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower().strip()
    s = re.sub(r"[–—\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’","'").replace("“",'"').replace("”",'"')
    return s

def _split_ed(ed: str):
    parts = re.split(DELIM_REGEX, _norm(ed), maxsplit=1)
    cat = parts[0].strip() if parts else ""
    sub = parts[1].strip() if len(parts) > 1 else ""
    return cat, sub

# ---------- registry I/O ----------
def load_registry(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version":"ED-Normaliser.v3","created":datetime.utcnow().isoformat(),
            "last_updated":None,"themes":{}}

def save_registry(reg: dict, path: str):
    reg["last_updated"] = datetime.utcnow().isoformat()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)

# ---------- Day 1: build canonicals by symmetric NN (god-tier compression) ----------
def day1_build_canonicals(df: pd.DataFrame, thresh_cat: float, thresh_sub: float):
    """
    Returns: registry dict, mapping df with canonical_experience_driver
    """
    reg = load_registry("<memory>")  # temporary
    out = df.copy()
    out["canonical_experience_driver"] = ""

    # theme-scope
    for theme, tdf in df.groupby(df["theme"].apply(_norm)):
        eds = sorted({f"{_split_ed(ed)[0]} → {_split_ed(ed)[1]}" for ed in tdf["experience_driver"].dropna().map(_norm)})
        if not eds:
            continue

        parts = {ed: _split_ed(ed) for ed in eds}
        # nearest neighbor dict
        nn = {}
        for i, edi in enumerate(eds):
            c1, s1 = parts[edi]
            best_j, best_pair, best_c, best_s = None, -1.0, 0.0, 0.0
            for j, edj in enumerate(eds):
                if i == j: continue
                c2, s2 = parts[edj]
                cs, ss = _sim(c1,c2), _sim(s1,s2)
                pair = min(cs, ss)
                if pair > best_pair:
                    best_j, best_pair, best_c, best_s = j, pair, cs, ss
            nn[i] = (best_j, best_c, best_s)

        # union by symmetric NN + thresholds
        parent = {ed: ed for ed in eds}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        for i, edi in enumerate(eds):
            j, c_ij, s_ij = nn[i]
            if j is None: continue
            edj = eds[j]
            j_back, c_ji, s_ji = nn[j]
            symmetric = (j_back == i)
            cat_ok = min(c_ij, c_ji) >= thresh_cat
            sub_ok = min(s_ij, s_ji) >= thresh_sub
            if symmetric and cat_ok and sub_ok:
                union(edi, edj)

        # groups -> canonical = shortest then lexicographic
        groups = defaultdict(list)
        for ed in eds:
            groups[find(ed)].append(ed)

        theme_key = _norm(theme)
        reg["themes"].setdefault(theme_key, {"canonicals":{}})
        for root, members in groups.items():
            canon = sorted(members, key=lambda x: (len(x), x))[0]
            cat, sub = _split_ed(canon)
            reg["themes"][theme_key]["canonicals"][canon] = {
                "category": cat, "subcategory": sub, "aliases": sorted(set(members))
            }

        # write mapped canonicals to out
        ed_to_canon = {}
        for _, members in groups.items():
            canon = sorted(members, key=lambda x: (len(x), x))[0]
            for m in members: ed_to_canon[m] = canon

        mask = df["theme"].apply(_norm) == theme_key
        out.loc[mask, "canonical_experience_driver"] = (
            df.loc[mask, "experience_driver"]
              .map(lambda x: f"{_split_ed(x)[0]} → {_split_ed(x)[1]}")
              .map(lambda ed: ed_to_canon.get(ed, ed))
        )

    return reg, out

# ---------- Day N: compare to existing registry; merge or create ----------
def dayn_assign_to_registry(df: pd.DataFrame, reg: dict, thresh_cat: float, thresh_sub: float):
    out = df.copy()
    out["canonical_experience_driver"] = ""

    for idx, row in df.iterrows():
        theme_key = _norm(row.get("theme",""))
        raw_ed = row.get("experience_driver","")
        if not theme_key or not isinstance(raw_ed, str):
            continue
        cat, sub = _split_ed(raw_ed)
        candidate = f"{cat} → {sub}"

        # if theme absent -> create theme + canonical
        reg["themes"].setdefault(theme_key, {"canonicals":{}})
        canonicals = list(reg["themes"][theme_key]["canonicals"].keys())
        if not canonicals:
            reg["themes"][theme_key]["canonicals"][candidate] = {
                "category": cat, "subcategory": sub, "aliases":[candidate]
            }
            out.at[idx, "canonical_experience_driver"] = candidate
            continue

        # pick best canonical by min(cat_sim, sub_sim)
        best, best_pair = None, -1.0
        for can in canonicals:
            c2, s2 = _split_ed(can)
            cs, ss = _sim(cat, c2), _sim(sub, s2)
            pair = min(cs, ss)
            if pair > best_pair:
                best, best_pair = can, pair

        if best and best_pair >= min(thresh_cat, thresh_sub):
            # merge into best canonical
            out.at[idx, "canonical_experience_driver"] = best
            if candidate not in reg["themes"][theme_key]["canonicals"][best]["aliases"]:
                reg["themes"][theme_key]["canonicals"][best]["aliases"].append(candidate)
        else:
            # create new canonical
            reg["themes"][theme_key]["canonicals"][candidate] = {
                "category": cat, "subcategory": sub, "aliases":[candidate]
            }
            out.at[idx, "canonical_experience_driver"] = candidate

    return reg, out

# ---------- CLI runner ----------
def run(input_csv: str, outdir: str, registry_path: str, thresh_cat: float, thresh_sub: float):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Day1 if no registry; DayN otherwise
    reg_exists = os.path.exists(registry_path)
    if not reg_exists:
        reg, mapped = day1_build_canonicals(df, thresh_cat, thresh_sub)
    else:
        reg = load_registry(registry_path)
        reg, mapped = dayn_assign_to_registry(df, reg, thresh_cat, thresh_sub)

    # save registry + artifacts
    save_registry(reg, registry_path)

    mapped_path = os.path.join(outdir, "dataset_with_canonical_ed.csv")
    uniq_path   = os.path.join(outdir, "unique_ed_report.csv")
    reg_path    = registry_path

    mapped.to_csv(mapped_path, index=False)
    uniq = mapped[["theme","canonical_experience_driver"]].drop_duplicates().copy()
    uniq[["canonical_category","canonical_subcategory"]] = (
        uniq["canonical_experience_driver"].apply(lambda x: pd.Series(_split_ed(x)))
    )
    uniq.to_csv(uniq_path, index=False)

    return mapped_path, uniq_path, reg_path

def main():
    ap = argparse.ArgumentParser(description="ED Normaliser — Day1/DayN + God-tier Compression")
    ap.add_argument("--input", required=True, help="CSV with columns: theme, experience_driver")
    ap.add_argument("--outdir", default="ed_normaliser_outputs")
    ap.add_argument("--registry", default="registry/ed_registry.json")
    ap.add_argument("--thresh_cat", type=float, default=THRESH_CAT)
    ap.add_argument("--thresh_sub", type=float, default=THRESH_SUB)
    args = ap.parse_args()

    global THRESH_CAT, THRESH_SUB
    THRESH_CAT, THRESH_SUB = args.thresh_cat, args.thresh_sub

    mapped, uniq, reg = run(args.input, args.outdir, args.registry, THRESH_CAT, THRESH_SUB)
    print("Artifacts:")
    print(" -", mapped)
    print(" -", uniq)
    print(" -", reg)

if __name__ == "__main__":
    main()
