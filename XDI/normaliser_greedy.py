# normaliser_greedy_global.py
# Global (theme-agnostic) ED normaliser:
# - Compares ALL Experience Drivers to each other (ignores theme)
# - Greedy two-loop clubbing using weighted-Jaccard over simple tokens
# - Subcategory (right of the arrow) gets more weight
# - Picks the medoid phrase as the canonical label
# - Writes a normalized CSV for XDI + a mapping CSV of what was clubbed

from __future__ import annotations
import os, re, json, hashlib, unicodedata
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# ======== EDIT THESE PATHS ========
INPUT  = r"data\retail_analytics_flattened_v2.csv"     # your parsed/cleaned CSV
OUTPUT = r"data\retail_analytics_normalised_v2.csv"   # normalised CSV for XDI
MAP    = r"data\retail_analytics_map_v2.csv"      # report of cluster → members

# Column names
ED_COL = "experience_driver"    # if yours differs, change here

# Parameters
THRESHOLD  = 0.66   # higher merges more; 0.50 merges “Product Freshness” & “Freshness”
WEIGHT_SUB = 2.0    # subcategory tokens weight vs category tokens
MINSIZE    = 1      # keep clusters even if size==1 (no forced merges)

# ======== TEXT UTILITIES ========
def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s))

def _clean_arrows(s: str) -> str:
    s = s.replace("->", "→").replace("-->", "→").replace("—>", "→")
    s = re.sub(r"\s*→\s*", " → ", s)
    return s

def _strip_zwc(s: str) -> str:
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", s)

def canon_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = _nfkc(str(s))
    s = _strip_zwc(s)
    s = _clean_arrows(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_ed_parts(ed: str) -> Tuple[str, str]:
    ed = canon_text(ed)
    parts = [p.strip() for p in ed.split("→", 1)]
    cat = parts[0] if parts else ""
    sub = parts[1] if len(parts) > 1 else ""
    return cat, sub

def tokens(s: str) -> List[str]:
    s = _nfkc(s).lower()
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return [t for t in s.split() if t]

def doc_weights(ed: str, weight_sub: float = WEIGHT_SUB) -> Dict[str, float]:
    cat, sub = split_ed_parts(ed)
    tw: Dict[str, float] = {}
    for t in tokens(cat):
        tw[t] = tw.get(t, 0.0) + 1.0
    for t in tokens(sub):
        tw[t] = tw.get(t, 0.0) + float(weight_sub)
    return tw

def weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k,0.0), b.get(k,0.0)) for k in keys)
    union = sum(max(a.get(k,0.0), b.get(k,0.0)) for k in keys)
    return 0.0 if union == 0 else inter / union

def medoid_index(indices: List[int], docs: List[Dict[str,float]]) -> int:
    if len(indices) == 1:
        return indices[0]
    best_i, best_sim = indices[0], -1.0
    for i in indices:
        s = 0.0
        for j in indices:
            if i == j: 
                continue
            s += weighted_jaccard(docs[i], docs[j])
        avg = s / max(1, len(indices)-1)
        if avg > best_sim:
            best_sim = avg; best_i = i
    return best_i

def stable_id(label: int) -> str:
    raw = f"GLOBAL::{label}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

# ======== GREEDY GLOBAL CLUBBING ========
def greedy_cluster_global(eds: List[str], threshold: float = THRESHOLD, weight_sub: float = WEIGHT_SUB) -> Tuple[List[List[int]], List[Dict[str,float]]]:
    """
    Returns (clusters, doc_vectors) for the unique ED list (global).
    Greedy: take ed[i], compare to remaining, club if sim >= threshold.
    """
    docs = [doc_weights(ed, weight_sub=weight_sub) for ed in eds]
    n = len(eds)
    assigned = [False]*n
    clusters: List[List[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i+1, n):
            if assigned[j]:
                continue
            if weighted_jaccard(docs[i], docs[j]) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters, docs

# add this helper near stable_id()
def stable_id_from_label(label_text: str, scope: str = "GLOBAL") -> str:
    raw = f"{scope}::{canon_text(label_text)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

# ======== MAIN ========
def normalize_file(input_path: str = INPUT,
                   output_path: str = OUTPUT,
                   map_path: str = MAP,
                   ed_col: str = ED_COL,
                   threshold: float = THRESHOLD,
                   weight_sub: float = WEIGHT_SUB,
                   minsize: int = MINSIZE) -> dict:
    df = pd.read_csv(input_path, encoding="utf-8").copy()
    if ed_col not in df.columns:
        raise ValueError(f"Missing required column: {ed_col}")

    # Clean ED text
    df[ed_col] = df[ed_col].map(canon_text)

    # Build global unique ED list (ignore theme entirely)
    uniq = df[[ed_col]].dropna().drop_duplicates().reset_index(drop=True)
    phrases = uniq[ed_col].tolist()
    if not phrases:
        raise ValueError("No Experience Driver values found.")

    clusters, docs = greedy_cluster_global(phrases, threshold=threshold, weight_sub=weight_sub)

    # Assign canonicals
    out_rows = []
    map_rows = []
    for lbl, idxs in enumerate(clusters):
        if len(idxs) < minsize:
            for i in idxs:
                canon = phrases[i]
                norm_id = stable_id_from_label(canon)
                mask = (df[ed_col] == phrases[i])
                out_rows.append((mask, canon, norm_id, 1))
                map_rows.append({
                    "cluster_label": f"{lbl}.{i}",
                    "ed_canonical": canon,
                    "size": 1,
                    "members": json.dumps([phrases[i]]),
                    "backend": "greedy-global",
                    "threshold": threshold,
                    "weight_sub": weight_sub,
                })
            continue

        med = medoid_index(idxs, docs)
        canon = phrases[med]
        members = [phrases[i] for i in idxs]
        norm_id = stable_id_from_label(canon)

        mask = df[ed_col].isin(members)
        out_rows.append((mask, canon, norm_id, len(idxs)))
        map_rows.append({
            "cluster_label": lbl,
            "ed_canonical": canon,
            "size": len(idxs),
            "members": json.dumps(members),
            "backend": "greedy-global",
            "threshold": threshold,
            "weight_sub": weight_sub,
        })

    # Materialize outputs
    df["experience_driver_norm"] = df[ed_col]
    df["ed_norm_id"] = ""
    df["ed_norm_backend"] = "greedy-global"
    df["ed_norm_threshold"] = threshold
    df["ed_norm_size"] = 1

    for mask, canon, norm_id, size in out_rows:
        df.loc[mask, "experience_driver_norm"] = canon
        df.loc[mask, "ed_norm_id"] = norm_id
        df.loc[mask, "ed_norm_size"] = size

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(map_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    pd.DataFrame(map_rows, columns=[
        "cluster_label","ed_canonical","size","members","backend","threshold","weight_sub"
    ]).to_csv(map_path, index=False, encoding="utf-8")

    return {
        "unique_EDs_before": int(df[ed_col].nunique(dropna=True)),
        "unique_EDs_after": int(df["experience_driver_norm"].nunique(dropna=True)),
        "output_path": output_path,
        "map_path": map_path,
        "backend": "greedy-global",
        "threshold": threshold,
        "weight_sub": weight_sub
    }

if __name__ == "__main__":
    stats = normalize_file()
    print("\n==== Greedy Global Normaliser Complete ====")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("==========================================\n")
