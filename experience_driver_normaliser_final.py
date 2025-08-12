
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED Normaliser – Component-Locked, Integrity-First
-------------------------------------------------
- Treats Experience Driver (ED) as infrastructure: Theme → (Category → Subcategory) → Entity Name
- Clubs only true mechanism-equivalents inside the same Theme
- Blocks merges across contrastive heads and qualifier conflicts
- Produces:
    1) club_candidates.csv (pairwise decisions + reasons)
    2) unique_ed_report.csv (final canonical EDs + semantic_stability)
    3) canonical_registry.json (per theme, with aliases)
    4) explosion_stats.txt
    5) dataset_with_canonical_ed.csv (original rows + canonical_experience_driver)
Dependencies: pandas, numpy (RapidFuzz optional; falls back to difflib)
"""

import os
import re
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Optional: RapidFuzz (faster/better similarity). Falls back to difflib if missing.
try:
    from rapidfuzz import fuzz
    def _lex_sim(a: str, b: str) -> float:
        # Convert 0..100 to 0..1
        return fuzz.token_sort_ratio(a, b) / 100.0
except Exception:
    import difflib
    def _lex_sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


# ------------------------- Config (edit if needed) -------------------------

THRESH_CAT = 0.90   # category similarity threshold (0..1)
THRESH_SUB = 0.90   # subcategory similarity threshold (0..1)

# Pairs that must NOT be merged (semantic contrasts)
CONTRASTIVE = [
    ("visibility","accuracy"),
    ("availability","allocation"),
    ("speed","reliability"),
    ("timing","slotting"),
    ("substitution","availability"),
    ("fees","pricing"),
    ("freeze","unfreeze"),
    ("authorization","authentication")
]

# Qualifiers that change mechanics; if present in one ED but not the other -> reject
QUALIFIERS = {"real-time","batch","scheduled","same-day","instant","async","offline","online"}

# Mild synonym/morphology canon (safe set only)
SYN_MAP = {
    "timeliness": "timing",
    "functionalities": "functionality",
    "availabilities": "availability",
}

# --------------------------------------------------------------------------


def norm_text(s: str) -> str:
    """Lowercase, strip, normalize whitespace and hyphens, and apply mild canon."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower().strip()
    s = re.sub(r"[–—\-]+", " ", s)           # hyphens/dashes -> space
    s = re.sub(r"\s+", " ", s)               # collapse spaces
    s = s.replace("’", "'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\breal time\b", "real-time", s)
    # apply SYN_MAP
    tokens = s.split(" ")
    tokens = [SYN_MAP.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def split_ed(ed: str):
    """Split 'Category → Subcategory' into (cat, sub)."""
    if not isinstance(ed, str):
        ed = "" if pd.isna(ed) else str(ed)
    parts = [p.strip() for p in str(ed).split("→")]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def contains_any(s: str, tokens: set) -> set:
    words = set(re.findall(r"[a-z0-9\-]+", s.lower()))
    return words & tokens


def build_theme_ed_index(df: pd.DataFrame):
    """Build unique ED keys per normalized theme."""
    work = df.copy()
    work["theme_norm"] = work["theme"].apply(norm_text)
    work["ed_raw_norm"] = work["experience_driver"].apply(norm_text)
    work[["cat_raw","subcat_raw"]] = work["ed_raw_norm"].apply(lambda x: pd.Series(split_ed(x)))

    # heads (post-canon of individual parts only)
    work["cat_head"] = work["cat_raw"]
    work["sub_head"] = work["subcat_raw"]

    eds_by_theme = defaultdict(list)
    for _, r in work.iterrows():
        theme = r["theme_norm"]
        cat = r["cat_head"]
        sub = r["sub_head"]
        ed_key = f"{cat} → {sub}".strip()
        eds_by_theme[theme].append(ed_key)

    for t in eds_by_theme:
        eds_by_theme[t] = sorted(list(set(eds_by_theme[t])))

    return work, eds_by_theme


def decide_clubs(eds_by_theme):
    """Pairwise compare EDs inside each theme and decide clubbing."""
    canonical_map = {}
    club_records = []

    for theme, ed_list in eds_by_theme.items():
        # init
        for ed in ed_list:
            canonical_map[(theme, ed)] = ed

        parts = {ed: tuple(split_ed(ed)) for ed in ed_list}

        for i in range(len(ed_list)):
            for j in range(i+1, len(ed_list)):
                ed1 = ed_list[i]; ed2 = ed_list[j]
                c1, s1 = parts[ed1]
                c2, s2 = parts[ed2]

                # Qualifier block
                q1 = contains_any(ed1, QUALIFIERS)
                q2 = contains_any(ed2, QUALIFIERS)
                if q1 != q2:
                    club_records.append({
                        "theme": theme, "ed_1": ed1, "ed_2": ed2,
                        "cat_sim": _lex_sim(c1,c2), "sub_sim": _lex_sim(s1,s2),
                        "qualifier_conflict": True,
                        "contrastive_conflict": False,
                        "decision": "REJECT_QUALIFIER_CONFLICT"
                    })
                    continue

                # Contrastive block
                contrastive_hit = False
                for a,b in CONTRASTIVE:
                    if a in ed1 and b in ed2: contrastive_hit=True; break
                    if b in ed1 and a in ed2: contrastive_hit=True; break
                if contrastive_hit:
                    club_records.append({
                        "theme": theme, "ed_1": ed1, "ed_2": ed2,
                        "cat_sim": _lex_sim(c1,c2), "sub_sim": _lex_sim(s1,s2),
                        "qualifier_conflict": False,
                        "contrastive_conflict": True,
                        "decision": "REJECT_CONTRASTIVE"
                    })
                    continue

                # Similarities
                c_sim = _lex_sim(c1, c2)
                s_sim = _lex_sim(s1, s2)

                if c_sim >= THRESH_CAT and s_sim >= THRESH_SUB:
                    canon = min(ed1, ed2, key=lambda x: (len(x), x))
                    canonical_map[(theme, ed1)] = canon
                    canonical_map[(theme, ed2)] = canon
                    decision = "ACCEPT"
                else:
                    decision = "REJECT_LOW_SIM"

                club_records.append({
                    "theme": theme,
                    "ed_1": ed1, "ed_2": ed2,
                    "cat_sim": round(c_sim,4),
                    "sub_sim": round(s_sim,4),
                    "qualifier_conflict": False,
                    "contrastive_conflict": False,
                    "decision": decision
                })

    return canonical_map, pd.DataFrame(club_records)


def build_registry(eds_by_theme, canonical_map):
    """Create per-theme canonical registry with alias groups."""
    registry = {
        "version": "ED-Normaliser.v1",
        "generated_at": datetime.utcnow().isoformat(),
        "themes": {}
    }

    for theme, ed_list in eds_by_theme.items():
        groups = defaultdict(list)
        for ed in ed_list:
            canon = canonical_map[(theme, ed)]
            groups[canon].append(ed)

        theme_entry = {"experience_drivers": []}
        for canon, aliases in groups.items():
            cat, sub = split_ed(canon)
            theme_entry["experience_drivers"].append({
                "canonical_experience_driver": canon,
                "canonical_category": cat,
                "canonical_subcategory": sub,
                "aliases": sorted(list(set(aliases)))
            })
        registry["themes"][theme] = theme_entry

    return registry


def apply_canonical(work: pd.DataFrame, canonical_map: dict) -> pd.DataFrame:
    """Map each row to its canonical ED (theme-scoped)."""
    def map_row(r):
        theme = norm_text(r["theme"])
        cat, sub = split_ed(norm_text(r["experience_driver"]))
        ed = f"{cat} → {sub}".strip()
        return canonical_map.get((theme, ed), ed)

    out = work.copy()
    out["canonical_experience_driver"] = out.apply(map_row, axis=1)
    return out


def compute_unique_report(out_df: pd.DataFrame) -> pd.DataFrame:
    """Unique canonical EDs per theme + 'semantic_stability' proxy score."""
    unique_rows = out_df[["theme","canonical_experience_driver"]].drop_duplicates().copy()
    unique_rows["theme_norm"] = unique_rows["theme"].apply(norm_text)

    # split parts
    parts = unique_rows["canonical_experience_driver"].apply(lambda x: pd.Series(split_ed(x)))
    unique_rows["cat"] = parts[0]
    unique_rows["sub"] = parts[1]

    # stability
    stabilities = []
    for theme, grp in unique_rows.groupby("theme_norm"):
        eds = grp["canonical_experience_driver"].tolist()
        pts = [split_ed(e) for e in eds]
        vals = []
        for i, e in enumerate(eds):
            c1, s1 = pts[i]
            best = 0.0
            for j, f in enumerate(eds):
                if i==j: continue
                c2, s2 = pts[j]
                best = max(best, 0.5*_lex_sim(c1,c2) + 0.5*_lex_sim(s1,s2))
            vals.append(1.0 - best)  # higher = more isolated/unique
        stabilities.extend(vals)

    unique_rows["semantic_stability"] = [round(v,4) for v in stabilities]
    cols = ["theme","canonical_experience_driver","cat","sub","semantic_stability"]
    return unique_rows[cols]


def write_stats(df_in: pd.DataFrame, df_out: pd.DataFrame, club_df: pd.DataFrame, path: str):
    orig_unique = df_in["experience_driver"].nunique()
    canon_unique = df_out["canonical_experience_driver"].nunique()
    compression_ratio = orig_unique / canon_unique if canon_unique else 0.0
    needs_review = club_df[club_df["decision"].str.startswith("REJECT")].shape[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
        f.write("="*50 + "\n")
        f.write(f"Original unique EDs: {orig_unique}\n")
        f.write(f"Canonical unique EDs: {canon_unique}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Pair decisions logged: {len(club_df)}\n")
        f.write(f"Pairs requiring review (rejected): {needs_review}\n")


def run(input_csv: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_csv)

    work, eds_by_theme = build_theme_ed_index(df)
    canonical_map, club_df = decide_clubs(eds_by_theme)
    registry = build_registry(eds_by_theme, canonical_map)
    out_df = apply_canonical(work, canonical_map)
    unique_report = compute_unique_report(out_df)

    # Save artifacts
    club_csv = os.path.join(outdir, "club_candidates.csv")
    unique_csv = os.path.join(outdir, "unique_ed_report.csv")
    canon_json = os.path.join(outdir, "canonical_registry.json")
    stats_txt = os.path.join(outdir, "explosion_stats.txt")
    canon_df_csv = os.path.join(outdir, "dataset_with_canonical_ed.csv")

    club_df.to_csv(club_csv, index=False)
    unique_report.to_csv(unique_csv, index=False)
    with open(canon_json, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    out_df_origcols = df.copy()
    out_df_origcols["canonical_experience_driver"] = out_df["canonical_experience_driver"]
    out_df_origcols.to_csv(canon_df_csv, index=False)
    write_stats(df, out_df_origcols, club_df, stats_txt)

    return club_csv, unique_csv, canon_json, stats_txt, canon_df_csv


def main():
    parser = argparse.ArgumentParser(description="Experience Driver Normaliser (component-locked)")
    parser.add_argument("--input", required=True, help="Input CSV with columns: theme, experience_driver, ...")
    parser.add_argument("--outdir", default="ed_normaliser_outputs", help="Output directory for artifacts")
    args = parser.parse_args()

    club_csv, unique_csv, canon_json, stats_txt, canon_df_csv = run(args.input, args.outdir)
    print("[✓] Canonicalization completed successfully")
    print("Artifacts:")
    print(" -", club_csv)
    print(" -", unique_csv)
    print(" -", canon_json)
    print(" -", stats_txt)
    print(" -", canon_df_csv)


if __name__ == "__main__":
    main()
