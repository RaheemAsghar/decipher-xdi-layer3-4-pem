#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOD-TIER Experience Driver Normaliser
-------------------------------------
- Zero curated ED rules (no contrastives/qualifiers/synonyms).
- Theme-scoped, component-aware (Category & Subcategory).
- Day 1: build canonicals via symmetric top-1 NN + thresholds.
- Day N: match each ED against existing theme canonicals (shortlisted via on-disk component indexes),
         merge if thresholds pass; else create a new canonical.
- Per-vertical thresholds from JSON (and optional per-theme overrides saved in registry).
- High-perf extras: batch processing, preallocated columns, on-disk component indexes with rebuild-on-mutation,
  registry telemetry (aliases, frequency, first_seen/last_seen), logging, reports.

Artifacts:
- <outdir>/dataset_with_canonical_ed.csv
- <outdir>/unique_ed_report.csv
- <registry>.json (themes → canonicals → aliases, metadata)
- <outdir>/explosion_stats.txt
"""

import os, re, json, argparse, logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, List
import pandas as pd

# -------------------- Similarity (RapidFuzz → difflib) --------------------
try:
    from rapidfuzz import fuzz
    def _sim(a: str, b: str) -> float:
        return fuzz.token_sort_ratio(a, b) / 100.0
except Exception:
    import difflib
    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

# -------------------- Neutral text utils (no curated semantics) --------------------
DELIM_REGEX = r"\s*(?:→|->|>|/|:)\s*"  # multiple delimiters accepted

def _norm(s):
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower().strip()
    s = re.sub(r"[–—\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’","'").replace("“",'"').replace("”",'"')
    return s

def _split_ed(ed: str) -> Tuple[str, str]:
    parts = re.split(DELIM_REGEX, _norm(ed), maxsplit=1)
    cat = parts[0].strip() if parts else ""
    sub = parts[1].strip() if len(parts) > 1 else ""
    return cat, sub

# -------------------- Threshold config --------------------
def load_threshold_map(path="thresholds.json") -> Dict:
    base = {"general": {"category": 0.90, "subcategory": 0.90}}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    base.update(data)
        except Exception:
            pass
    return base

def resolve_thresholds(theme_key: str, vertical: str, thr_map: dict, registry: dict) -> Tuple[float,float]:
    # 1) registry per-theme override
    t = (
        registry.get("thresholds", {})
                .get("themes", {})
                .get(theme_key)
    )
    if isinstance(t, dict) and {"category","subcategory"} <= set(t):
        return float(t["category"]), float(t["subcategory"])
    # 2) per-vertical from config
    v = thr_map.get(vertical) or thr_map.get("general", {})
    return float(v.get("category", 0.90)), float(v.get("subcategory", 0.90))

# -------------------- Registry I/O + component index --------------------
def load_registry(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reg = json.load(f)
                reg.setdefault("themes", {})
                return reg
        except Exception:
            pass
    return {"version":"ED-Normaliser.v4",
            "created":datetime.utcnow().isoformat(),
            "last_updated":None,
            "themes":{},
            "thresholds":{"themes":{}},
            "_component_index":{}, "_mutations":0}

def save_registry(reg: dict, path: str):
    reg["last_updated"] = datetime.utcnow().isoformat()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)

def build_theme_component_index(reg: dict) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    idx = {}
    for theme, tdata in reg.get("themes", {}).items():
        cats, subs = {}, {}
        for can in tdata.get("canonicals", {}).keys():
            c, s = _split_ed(can)
            cn, sn = _norm(c), _norm(s)
            cats.setdefault(cn, set()).add(can)
            subs.setdefault(sn, set()).add(can)
        idx[theme] = {"cats": {k: sorted(v) for k,v in cats.items()},
                      "subs": {k: sorted(v) for k,v in subs.items()}}
    return idx

def maybe_rebuild_index(reg: dict, rebuild_threshold: int):
    reg.setdefault("_mutations", 0)
    if reg["_mutations"] >= rebuild_threshold:
        reg["_component_index"] = build_theme_component_index(reg)
        reg["_mutations"] = 0

# -------------------- Day 1: symmetric NN compression (per theme) --------------------
class _UF:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def day1_build_canonicals(df: pd.DataFrame, reg: dict, get_thr):
    out = df.copy()
    out["canonical_experience_driver"] = ""

    for theme_key, tdf in df.groupby(df["theme"].apply(_norm)):
        eds = sorted({
            f"{_split_ed(ed)[0]} → {_split_ed(ed)[1]}"
            for ed in tdf["experience_driver"].dropna().map(_norm)
        })
        if not eds: continue

        parts = {ed: _split_ed(ed) for ed in eds}
        # nearest neighbor table
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

        uf = _UF(eds)
        thr_cat, thr_sub = get_thr(theme_key)
        for i, edi in enumerate(eds):
            j, c_ij, s_ij = nn[i]
            if j is None: continue
            edj = eds[j]
            j_back, c_ji, s_ji = nn[j]
            symmetric = (j_back == i)
            cat_ok = min(c_ij, c_ji) >= thr_cat
            sub_ok = min(s_ij, s_ji) >= thr_sub
            if symmetric and cat_ok and sub_ok:
                uf.union(edi, edj)

        groups = defaultdict(list)
        for ed in eds:
            groups[uf.find(ed)].append(ed)

        reg["themes"].setdefault(theme_key, {"canonicals":{}})
        for root, members in groups.items():
            canon = sorted(members, key=lambda x: (len(x), x))[0]
            cat, sub = _split_ed(canon)
            meta = reg["themes"][theme_key]["canonicals"].setdefault(canon, {
                "category": cat, "subcategory": sub, "aliases": [], "frequency":0
            })
            # add all members as aliases (dedup)
            for m in members:
                if m not in meta["aliases"]:
                    meta["aliases"].append(m)
            # telemetry
            today = datetime.utcnow().date().isoformat()
            meta["first_seen"] = meta.get("first_seen", today)
            meta["last_seen"]  = today
            meta["frequency"]  = int(meta.get("frequency", 0)) + len(members)
            reg["_mutations"] += 1

        # write mapped canonicals
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

# -------------------- Day N: shortlist via component index, then match --------------------
def _touch_registry_alias(reg: dict, theme_key: str, canonical: str, variant: str):
    meta = reg["themes"][theme_key]["canonicals"].setdefault(
        canonical,
        {"category": _split_ed(canonical)[0],
         "subcategory": _split_ed(canonical)[1],
         "aliases": [], "frequency":0}
    )
    if variant not in meta["aliases"]:
        meta["aliases"].append(variant)
    today = datetime.utcnow().date().isoformat()
    meta["first_seen"] = meta.get("first_seen", today)
    meta["last_seen"]  = today
    meta["frequency"]  = int(meta.get("frequency", 0)) + 1

def dayn_assign_to_registry(df: pd.DataFrame, reg: dict, get_thr, shortlist_backstop: int = 200):
    out = df.copy()
    out["canonical_experience_driver"] = ""
    out["match_score_min_pair"] = 0.0

    # ensure component index exists
    if not reg.get("_component_index"):
        reg["_component_index"] = build_theme_component_index(reg)

    for idx, row in df.iterrows():
        theme_key = _norm(row.get("theme",""))
        raw_ed = row.get("experience_driver","")
        if not theme_key or not isinstance(raw_ed, str):
            continue

        reg["themes"].setdefault(theme_key, {"canonicals":{}})
        canonicals = list(reg["themes"][theme_key]["canonicals"].keys())

        cat, sub = _split_ed(raw_ed)
        candidate = f"{cat} → {sub}"

        # no canonicals yet → create
        if not canonicals:
            reg["themes"][theme_key]["canonicals"][candidate] = {
                "category": cat, "subcategory": sub, "aliases":[candidate],
                "first_seen": datetime.utcnow().date().isoformat(),
                "last_seen":  datetime.utcnow().date().isoformat(),
                "frequency": 1
            }
            out.at[idx, "canonical_experience_driver"] = candidate
            out.at[idx, "match_score_min_pair"] = 1.0
            reg["_mutations"] += 1
            continue

        # shortlist with component index
        comp_idx = reg["_component_index"].get(theme_key, {"cats":{}, "subs":{}})
        cn, sn = _norm(cat), _norm(sub)
        shortlist = set()
        shortlist |= set(comp_idx.get("cats", {}).get(cn, []))
        shortlist |= set(comp_idx.get("subs", {}).get(sn, []))
        if not shortlist:
            # fallback to all canonicals, but cap to avoid O(N^2) blowups
            shortlist = set(canonicals[:shortlist_backstop])

        # pick best by pair score = min(cat_sim, sub_sim)
        best, best_pair = None, -1.0
        for can in shortlist:
            c2, s2 = _split_ed(can)
            cs, ss = _sim(cat, c2), _sim(sub, s2)
            pair = min(cs, ss)
            if pair > best_pair:
                best, best_pair = can, pair

        thr_cat, thr_sub = get_thr(theme_key)
        pass_pair = best_pair >= min(thr_cat, thr_sub)

        if best and pass_pair:
            out.at[idx, "canonical_experience_driver"] = best
            out.at[idx, "match_score_min_pair"] = round(best_pair, 4)
            _touch_registry_alias(reg, theme_key, best, candidate)
        else:
            # new canonical
            reg["themes"][theme_key]["canonicals"][candidate] = {
                "category": cat, "subcategory": sub, "aliases":[candidate],
                "first_seen": datetime.utcnow().date().isoformat(),
                "last_seen":  datetime.utcnow().date().isoformat(),
                "frequency": 1
            }
            out.at[idx, "canonical_experience_driver"] = candidate
            out.at[idx, "match_score_min_pair"] = 1.0
        reg["_mutations"] += 1

    return reg, out

# -------------------- Batch processing & reports --------------------
def write_stats(df_in, df_out, path):
    orig_unique = df_in["experience_driver"].nunique()
    canon_unique = df_out["canonical_experience_driver"].nunique()
    compression_ratio = (orig_unique / canon_unique) if canon_unique else 0.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
        f.write("="*50 + "\n")
        f.write(f"Original unique EDs: {orig_unique}\n")
        f.write(f"Canonical unique EDs: {canon_unique}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Total rows: {len(df_in)}\n")

def export_unique_report(mapped: pd.DataFrame, out_path: str):
    uniq = mapped[["theme","canonical_experience_driver"]].drop_duplicates().copy()
    uniq[["canonical_category","canonical_subcategory"]] = (
        uniq["canonical_experience_driver"].apply(lambda x: pd.Series(_split_ed(x)))
    )
    uniq.to_csv(out_path, index=False)

# -------------------- Runner --------------------
def run(input_csv: str, outdir: str, registry_path: str,
        vertical: str, thr_config: str,
        default_thresh_cat: float, default_thresh_sub: float,
        rebuild_threshold: int):

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(outdir, "normaliser.log")),
                  logging.StreamHandler()]
    )

    reg_exists = os.path.exists(registry_path)
    reg = load_registry(registry_path) if reg_exists else load_registry("<new>")

    thr_map = load_threshold_map(thr_config)

    def get_thr(theme_key: str):
        # allow CLI defaults to override if vertical missing
        cat, sub = resolve_thresholds(theme_key, vertical, thr_map, reg if reg_exists else {})
        if vertical not in thr_map:
            cat, sub = default_thresh_cat, default_thresh_sub
        return cat, sub

    if not reg_exists:
        logging.info("Day 1: building canonicals…")
        reg["_component_index"] = {}  # will be built after save
        reg, mapped = day1_build_canonicals(df, reg, get_thr)
    else:
        logging.info("Day N: assigning to existing registry…")
        reg, mapped = dayn_assign_to_registry(df, reg, get_thr)
        maybe_rebuild_index(reg, rebuild_threshold)

    # (Re)build component index if missing
    if not reg.get("_component_index"):
        reg["_component_index"] = build_theme_component_index(reg)

    # save artifacts
    save_registry(reg, registry_path)

    mapped_path = os.path.join(outdir, "dataset_with_canonical_ed.csv")
    uniq_path   = os.path.join(outdir, "unique_ed_report.csv")
    stats_path  = os.path.join(outdir, "explosion_stats.txt")

    mapped.to_csv(mapped_path, index=False)
    export_unique_report(mapped, uniq_path)
    write_stats(df, mapped, stats_path)

    return mapped_path, uniq_path, registry_path, stats_path

def main():
    ap = argparse.ArgumentParser(description="GOD-TIER ED Normaliser (Day1/DayN, zero curated rules)")
    ap.add_argument("--input", required=True, help="CSV with columns: theme, experience_driver")
    ap.add_argument("--outdir", default="ed_normaliser_outputs")
    ap.add_argument("--registry", default="registry/ed_registry.json")
    ap.add_argument("--vertical", default="general", help="Key in thresholds.json")
    ap.add_argument("--thr_config", default="thresholds.json", help="Path to thresholds map JSON")
    ap.add_argument("--thresh_cat", type=float, default=0.90, help="Default Category threshold (fallback)")
    ap.add_argument("--thresh_sub", type=float, default=0.90, help="Default Subcategory threshold (fallback)")
    ap.add_argument("--rebuild_threshold", type=int, default=100, help="Rebuild component index after N mutations")
    args = ap.parse_args()

    mapped, uniq, reg, stats = run(
        input_csv=args.input,
        outdir=args.outdir,
        registry_path=args.registry,
        vertical=args.vertical,
        thr_config=args.thr_config,
        default_thresh_cat=args.thresh_cat,
        default_thresh_sub=args.thresh_sub,
        rebuild_threshold=args.rebuild_threshold
    )
    print("Artifacts:")
    print(" -", mapped)
    print(" -", uniq)
    print(" -", reg)
    print(" -", stats)

if __name__ == "__main__":
    main()
