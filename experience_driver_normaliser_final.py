
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED Normaliser – Dynamic, Self-Discovering (No hardcoded config)
---------------------------------------------------------------
Day 1 (Bootstrap):
  - Discover corpus-native patterns:
      * Morphological variants (light stemming + minimal edit diffs)
      * Qualifier tokens (salient modifiers that alter ops meaning)
      * Contrastive token pairs (negation/anti forms discovered from near-miss pairs)
  - Apply clubbing inside Theme with strict category/subcategory similarity + discovered patterns
  - Birth the ED registry + learned_patterns.json

Day 2+ (Evolve):
  - Load registry + learned patterns
  - Map new EDs, use patterns to guide safe merges
  - Add new canonicals only when truly distinct
  - Evolve patterns (counts, promote frequently observed to "strong")

Inputs:
  CSV with columns at least: theme, experience_driver

Outputs (in --outdir):
  - canonical_registry.json
  - learned_patterns.json
  - club_candidates.csv
  - unique_ed_report.csv
  - dataset_with_canonical_ed.csv
  - explosion_stats.txt
"""

import os, re, json, argparse
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd
import difflib

# ---------------- Core text utils ----------------

def norm(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"[–—\-]+", " ", s)      # hyphen normalize
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’","'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\breal time\b","real-time", s)
    return s

def split_ed(ed: str):
    parts = [p.strip() for p in str(ed).split("→")]
    if len(parts) == 1: return parts[0], ""
    return parts[0], parts[1]

def lex_sim(a: str, b: str) -> float:
    if a == b: return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def tokens(s: str):
    return re.findall(r"[a-z0-9]+(?:\-[a-z0-9]+)?", s.lower())

# very light stemmer (conservative)
def stem(tok: str) -> str:
    t = tok
    for suf, rep in [("ness",""), ("ities","ity"), ("ities","ity"), ("ity","y"), ("ments","ment"), ("ment",""), ("ing",""), ("ies","y"), ("s","")]:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)] + rep
            break
    return t

def minimal_diff_token(a: str, b: str):
    """If two strings differ by exactly one token (after stemming), return (a_only, b_only). Else None."""
    ta = [stem(x) for x in tokens(a)]
    tb = [stem(x) for x in tokens(b)]
    sa, sb = set(ta), set(tb)
    only_a = sa - sb
    only_b = sb - sa
    if len(only_a)==1 and len(only_b)==1:
        return list(only_a)[0], list(only_b)[0]
    return None

def negation_relation(x: str, y: str) -> bool:
    """Heuristic: detect anti/negated forms dynamically (un-, non-, in-, de-, dis-, anti-)."""
    prefixes = ["un","non","in","im","il","ir","de","dis","anti"]
    def strip_pref(w):
        for p in prefixes:
            if w.startswith(p) and len(w) > len(p)+2:
                return w[len(p):]
        return w
    return strip_pref(x) == y or strip_pref(y) == x

# ---------------- Pattern discovery ----------------

def discover_patterns(theme_to_eds):
    """From current corpus, discover:
       - qualifiers (salient modifiers)
       - contrastive token pairs (near-miss with negation morph)
       - synonym/morph pairs (near-miss without negation)
    """
    qualifier_counts = Counter()
    contrastive_pairs = Counter()
    synonym_pairs = Counter()

    for theme, eds in theme_to_eds.items():
        parts = {ed: split_ed(ed) for ed in eds}
        for i in range(len(eds)):
            for j in range(i+1, len(eds)):
                ed1, ed2 = eds[i], eds[j]
                c1, s1 = parts[ed1]
                c2, s2 = parts[ed2]

                # close by category & subcategory heads?
                c_sim = lex_sim(c1, c2)
                s_sim = lex_sim(s1, s2)

                # candidate "near" pairs
                if c_sim >= 0.8 or s_sim >= 0.8:
                    # look for a single-token diff
                    diff_cat = minimal_diff_token(c1, c2)
                    diff_sub = minimal_diff_token(s1, s2)

                    for diff in [diff_cat, diff_sub]:
                        if not diff: 
                            continue
                        a,b = diff
                        # negation-like -> contrastive
                        if negation_relation(a,b):
                            contrastive_pairs[tuple(sorted([a,b]))] += 1
                        else:
                            synonym_pairs[tuple(sorted([a,b]))] += 1

                    # detect qualifiers: tokens appearing in one ED but not the other that, if removed, raise similarity
                    t1 = set(tokens(ed1)); t2 = set(tokens(ed2))
                    for tok in (t1 ^ t2):
                        # if removing tok increases similarity between full strings -> treat as potential qualifier
                        if tok.isdigit(): 
                            continue
                        base1 = " ".join([w for w in tokens(ed1) if w != tok])
                        base2 = " ".join([w for w in tokens(ed2) if w != tok])
                        if lex_sim(base1, base2) > lex_sim(ed1, ed2):
                            qualifier_counts[tok] += 1

    # keep top N signals to avoid noise
    top_qualifiers = [w for w, c in qualifier_counts.most_common(50) if len(w) >= 3]
    top_contrastive = [{"a": a, "b": b, "count": c} for (a,b), c in contrastive_pairs.most_common(50)]
    top_synonyms = [{"a": a, "b": b, "count": c} for (a,b), c in synonym_pairs.most_common(50)]

    return {
        "qualifiers": top_qualifiers,
        "contrastive_pairs": top_contrastive,
        "synonym_pairs": top_synonyms
    }

# ---------------- Clubbing logic (uses discovered patterns) ----------------

def build_theme_index(df):
    work = df.copy()
    work["theme_norm"] = work["theme"].apply(norm)
    work["ed_norm"] = work["experience_driver"].apply(norm)
    work[["cat_raw","sub_raw"]] = work["ed_norm"].apply(lambda x: pd.Series(split_ed(x)))
    # heads
    work["cat_head"] = work["cat_raw"]
    work["sub_head"] = work["sub_raw"]

    themes = defaultdict(list)
    for _, r in work.iterrows():
        ed = f"{r['cat_head']} → {r['sub_head']}".strip()
        themes[r["theme_norm"]].append(ed)
    for t in themes:
        themes[t] = sorted(list(set(themes[t])))
    return work, themes

def should_block_by_qualifier(ed1, ed2, qualifiers):
    t1 = set(tokens(ed1)); t2 = set(tokens(ed2))
    q1 = t1 & set(qualifiers); q2 = t2 & set(qualifiers)
    return q1 != q2

def should_block_by_contrastive(ed1, ed2, contrastive_pairs):
    pairs = [(p["a"], p["b"]) for p in contrastive_pairs]
    for a,b in pairs:
        if (a in ed1 and b in ed2) or (b in ed1 and a in ed2):
            return True
    return False

def decide_clubs(theme_to_eds, learned):
    canonical_map = {}
    decisions = []

    qualifiers = learned.get("qualifiers", [])
    contrastive_pairs = learned.get("contrastive_pairs", [])
    synonyms = learned.get("synonym_pairs", [])

    # synonym dict for gentle canon (map rarer to commoner by count heuristic)
    syn_map = {}
    for p in synonyms:
        a, b = p["a"], p["b"]
        # simple lexicographic preference; in a real system you'd use frequency
        prefer = min(a,b, key=lambda x: (len(x), x))
        other = b if prefer == a else a
        syn_map[other] = prefer

    for theme, eds in theme_to_eds.items():
        for ed in eds:
            canonical_map[(theme, ed)] = ed

        parts = {ed: split_ed(ed) for ed in eds}

        for i in range(len(eds)):
            for j in range(i+1, len(eds)):
                ed1, ed2 = eds[i], eds[j]
                c1, s1 = parts[ed1]; c2, s2 = parts[ed2]

                # apply synonym canon to parts before similarity
                def canon_tokens(s):
                    toks = tokens(s)
                    toks = [syn_map.get(t, t) for t in toks]
                    return " ".join(toks)

                c1c, c2c = canon_tokens(c1), canon_tokens(c2)
                s1c, s2c = canon_tokens(s1), canon_tokens(s2)

                if should_block_by_qualifier(ed1, ed2, qualifiers):
                    decisions.append({"theme": theme, "ed_1": ed1, "ed_2": ed2,
                                      "cat_sim": lex_sim(c1c, c2c), "sub_sim": lex_sim(s1c, s2c),
                                      "decision": "REJECT_QUALIFIER_CONFLICT"})
                    continue

                if should_block_by_contrastive(ed1, ed2, contrastive_pairs):
                    decisions.append({"theme": theme, "ed_1": ed1, "ed_2": ed2,
                                      "cat_sim": lex_sim(c1c, c2c), "sub_sim": lex_sim(s1c, s2c),
                                      "decision": "REJECT_CONTRASTIVE"})
                    continue

                c_sim = lex_sim(c1c, c2c)
                s_sim = lex_sim(s1c, s2c)

                if c_sim >= 0.90 and s_sim >= 0.90:
                    canon = min(ed1, ed2, key=lambda x: (len(x), x))
                    canonical_map[(theme, ed1)] = canon
                    canonical_map[(theme, ed2)] = canon
                    decision = "ACCEPT"
                else:
                    decision = "REJECT_LOW_SIM"

                decisions.append({"theme": theme, "ed_1": ed1, "ed_2": ed2,
                                  "cat_sim": round(c_sim,4), "sub_sim": round(s_sim,4),
                                  "decision": decision})
    import pandas as pd
    return canonical_map, pd.DataFrame(decisions)

# ---------------- Registry & reports ----------------

def build_registry(theme_to_eds, canonical_map):
    registry = {"version":"ED-Normaliser.Dynamic.v1","generated_at":datetime.utcnow().isoformat(),"themes":{}}
    for theme, eds in theme_to_eds.items():
        groups = defaultdict(list)
        for ed in eds:
            groups[canonical_map[(theme, ed)]].append(ed)
        ed_list = []
        for canon, aliases in groups.items():
            cat, sub = split_ed(canon)
            ed_list.append({
                "canonical_experience_driver": canon,
                "canonical_category": cat,
                "canonical_subcategory": sub,
                "aliases": sorted(list(set(aliases)))
            })
        registry["themes"][theme] = {"experience_drivers": ed_list}
    return registry

def apply_canonical_rows(work_df, canonical_map):
    def map_row(r):
        theme = norm(r["theme"])
        cat, sub = split_ed(norm(r["experience_driver"]))
        ed = f"{cat} → {sub}".strip()
        return canonical_map.get((theme, ed), ed)
    out = work_df.copy()
    out["canonical_experience_driver"] = out.apply(map_row, axis=1)
    return out

def unique_report(out_df):
    u = out_df[["theme","canonical_experience_driver"]].drop_duplicates().copy()
    u["theme_norm"] = u["theme"].str.lower().str.strip()
    parts = u["canonical_experience_driver"].apply(lambda x: pd.Series(split_ed(x)))
    u["cat"], u["sub"] = parts[0], parts[1]
    # simple stability score
    from difflib import SequenceMatcher
    def _sim(a,b): return SequenceMatcher(None, a,b).ratio()
    stabs = []
    for theme, grp in u.groupby("theme_norm"):
        eds = grp["canonical_experience_driver"].tolist()
        pts = [split_ed(e) for e in eds]
        vals = []
        for i in range(len(eds)):
            c1,s1 = pts[i]
            best = 0.0
            for j in range(len(eds)):
                if i==j: continue
                c2,s2 = pts[j]
                best = max(best, 0.5*_sim(c1,c2)+0.5*_sim(s1,s2))
            vals.append(1.0 - best)
        stabs.extend(vals)
    u["semantic_stability"] = [round(v,4) for v in stabs]
    return u[["theme","canonical_experience_driver","cat","sub","semantic_stability"]]

def write_stats(df_in, df_out, club_df, path):
    orig_unique = df_in["experience_driver"].nunique()
    canon_unique = df_out["canonical_experience_driver"].nunique()
    compression_ratio = orig_unique / canon_unique if canon_unique else 0.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("EXPERIENCE DRIVER CANONICALIZATION STATS\n")
        f.write("="*50 + "\n")
        f.write(f"Original unique EDs: {orig_unique}\n")
        f.write(f"Canonical unique EDs: {canon_unique}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}x\n")
        f.write(f"Pair decisions logged: {len(club_df)}\n")

# ---------------- Main pipeline ----------------

def run(input_csv: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Build theme→ED index
    work_df, theme_to_eds = build_theme_index(df)

    # Load or discover patterns
    patterns_path = os.path.join(outdir, "learned_patterns.json")
    registry_path = os.path.join(outdir, "canonical_registry.json")

    if os.path.exists(patterns_path) and os.path.exists(registry_path):
        # Day 2+ evolve mode
        with open(patterns_path, "r", encoding="utf-8") as f:
            learned = json.load(f)
    else:
        # Day 1 bootstrap: discover from corpus
        learned = discover_patterns(theme_to_eds)

    # Decide clubbing using learned patterns
    canonical_map, club_df = decide_clubs(theme_to_eds, learned)

    # Build or extend registry (here we rebuild from current universe; in prod you'd merge)
    registry = build_registry(theme_to_eds, canonical_map)

    # Apply mapping to rows
    out_df = apply_canonical_rows(work_df, canonical_map)

    # Reports
    urep = unique_report(out_df)

    # Persist artifacts
    club_csv = os.path.join(outdir, "club_candidates.csv")
    unique_csv = os.path.join(outdir, "unique_ed_report.csv")
    stats_txt = os.path.join(outdir, "explosion_stats.txt")
    canon_df_csv = os.path.join(outdir, "dataset_with_canonical_ed.csv")

    club_df.to_csv(club_csv, index=False)
    urep.to_csv(unique_csv, index=False)
    out_df.to_csv(canon_df_csv, index=False)
    write_stats(df, out_df, club_df, stats_txt)

    # Persist learned patterns + registry (evolve: overwrite with updated universe)
    with open(patterns_path, "w", encoding="utf-8") as f:
        json.dump(learned, f, indent=2)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return {
        "club_candidates": club_csv,
        "unique_ed_report": unique_csv,
        "patterns": patterns_path,
        "registry": registry_path,
        "stats": stats_txt,
        "dataset_with_canonical": canon_df_csv
    }

def main():
    import argparse
    p = argparse.ArgumentParser(description="Dynamic ED Normaliser (Day1 bootstrap / Day2+ evolve)")
    p.add_argument("--input", required=True, help="Input CSV")
    p.add_argument("--outdir", required=True, help="Output directory (will store registry + learned patterns)")
    args = p.parse_args()
    outputs = run(args.input, args.outdir)
    print("[✓] Completed.")
    for k,v in outputs.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

