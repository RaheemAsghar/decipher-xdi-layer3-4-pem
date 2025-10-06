Got it. Here’s only what you asked for: **the minimal helpers** and the **OU traceability patch** (signature + OU ID + header & membership). Paste straight into your code.

```python
# ---------- helpers (minimal) ----------
import re, unicodedata, hashlib
from datetime import datetime
import pandas as pd

def _canon_txt(s: str) -> str:
    """Match parser canon: NFKC → strip ZWCs → arrow canon → space canon → lower."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("->", "→").replace("—>", "→").replace("-->", "→")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _sha(s: str, n: int) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]

def make_cluster_signature(preview_full: str) -> str:
    """Stable signature from the FULL (untruncated) preview."""
    return "clus:" + _sha(_canon_txt(preview_full), 16)

def make_ou_id(ed_id: str, feedback_type: str, opportunity_stream: str, cluster_signature: str) -> str:
    """Deterministic OU ID anchored to ED + FT + Stream + Signature."""
    parts = [str(ed_id or ""), str(feedback_type or ""), str(opportunity_stream or ""), str(cluster_signature or "")]
    return "ou:" + _sha("|".join(parts), 24)
```

```python
# ---------- inside your clustering loop (right where you have `preview`/`truncated_preview` and `grp`) ----------
# full (untruncated) preview → signature
full_preview = preview                      # use your existing `preview` BEFORE truncation
cluster_signature = make_cluster_signature(full_preview)

# anchors (from the group's first row; fallback to call-level args if needed)
first_row = grp.iloc[0]
ed_id_anchor = first_row.get("ed_id", "")
ft_anchor = first_row.get("feedback_type", str(feedback_type))
st_anchor = first_row.get("opportunity_stream", str(stream))

# deterministic OU id
ou_id = make_ou_id(ed_id_anchor, ft_anchor, st_anchor, cluster_signature)

# annotate rows for lineage
grp["cluster_signature"] = cluster_signature
grp["ou_id"] = ou_id

# write back to main df / store as you already do
df.update(grp)
cluster_store[group_id] = grp
```

```python
# ---------- AFTER the clustering loop: build minimal header + membership ----------
# Assumes: `cluster_store` (group_id -> grp), `full_composites` (group_id -> composite),
#          and (optionally) self._last_cluster_control with thresholds used.

ou_header_rows, ou_member_rows = [], []

for gid, comp in full_composites.items():
    any_row = cluster_store[gid].iloc[0]
    ou_header_rows.append({
        "ou_id": any_row["ou_id"],
        "cluster_signature": any_row["cluster_signature"],
        "ed_id": any_row.get("ed_id", ""),
        "experience_driver": any_row.get("experience_driver", ""),
        "feedback_type": any_row.get("feedback_type", ""),
        "opportunity_stream": any_row.get("opportunity_stream", ""),
        "cluster_theme_preview": comp["cluster_theme_preview"],   # your (possibly truncated) label
        "cluster_size": comp["cluster_size"],
        "cluster_cohesion": comp.get("cluster_cohesion"),
        "selection_status": "candidate",
        "ou_selection_timestamp": datetime.utcnow().isoformat() + "Z",
        # carry-through (if present on rows)
        "tenant_id": any_row.get("tenant_id", ""),
        "environment": any_row.get("environment", ""),
        "pipeline_run_id": any_row.get("pipeline_run_id", ""),
        # transparency (optional)
        "bcs_thr_start": getattr(self, "_last_cluster_control", {}).get("bcs_distance_threshold_start", None),
        "bcs_thr_used": getattr(self, "_last_cluster_control", {}).get("bcs_distance_threshold_used", None),
    })

    # one membership row per ED occurrence
    for ed_row_id in cluster_store[gid]["ed_row_id"].astype(str).tolist():
        ou_member_rows.append({"ou_id": any_row["ou_id"], "ed_row_id": ed_row_id})

ou_header_df = pd.DataFrame(ou_header_rows)       # -> xdi_ou_header
ou_membership_df = pd.DataFrame(ou_member_rows)   # -> xdi_ou_membership
```

**That’s all you need** for OU↔ED traceability:

* Stable `cluster_signature` from the **full** preview
* Deterministic `ou_id = f(ed_id, feedback_type, opportunity_stream, cluster_signature)`
* Rows annotated with `ou_id` + `cluster_signature`
* Lean `ou_header_df` and `ou_membership_df` artifacts for audit/replay.
