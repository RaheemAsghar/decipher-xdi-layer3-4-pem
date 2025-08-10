
# CSLI (Cross-Stream Learning Intelligence) – Python skeleton

```python
# csli.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
from datetime import datetime

# ---------- Data contracts (minimal) ----------
@dataclass
class OURecord:
    ou_id: str
    experience_driver: str
    opportunity_stream: str  # Fix/Optimize/Amplify/Innovate
    emotion: str             # e.g., "Agitation"
    problem_statement: str   # canonical paragraph
    activation_status: str   # "Activated" | "Pending"
    eri_score: Optional[float] = None
    rf_urgency_category: Optional[str] = None  # "Escalate" | "Monitor" | "Watch" ...

# IME accessor that your stack should already provide
class IME:
    def list_all_ous(self) -> List[OURecord]:
        raise NotImplementedError

# ---------- Lead-OU policy (choose any deterministic rule you like) ----------
def score_ou_for_lead(ou: OURecord) -> Tuple[int, float]:
    rf_rank = {"Escalate":3, "Monitor":2, "Watch":1}
    eri = ou.eri_score if ou.eri_score is not None else -999.0
    rf_score = rf_rank.get(ou.rf_urgency_category or "Watch", 1)
    return (rf_score, eri)  # higher better

# ---------- Clustering by semantic proximity ----------
def connected_components(adjacency: Dict[int, List[int]]) -> List[List[int]]:
    seen, comps = set(), []
    for node in adjacency.keys():
        if node in seen: continue
        comp, q = [], deque([node])
        seen.add(node)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps

def form_csli_groups(
    ime: IME,
    embed_fn: Callable[[List[str]], np.ndarray],  # returns (N, d) embeddings
    similarity_threshold: float = 0.82,
    now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    now = now or datetime.utcnow()
    ous = ime.list_all_ous()
    if not ous:
        return []

    texts = [ou.problem_statement for ou in ous]
    embs = embed_fn(texts)  # shape (N, d)
    sims = cosine_similarity(embs)  # (N, N)

    # build threshold graph
    N = len(ous)
    adj = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i+1, N):
            if sims[i, j] >= similarity_threshold:
                adj[i].append(j)
                adj[j].append(i)

    comps = connected_components(adj)
    outputs = []
    gid_counter = 1

    for comp in comps:
        if len(comp) < 2:
            continue  # only form CSLI groups for 2+ members

        members = [ous[i] for i in comp]
        # canonical problem statement = centroid text approximated by medoid
        sub_sims = sims[np.ix_(comp, comp)]
        medoid_idx = int(np.argmax(sub_sims.sum(axis=1)))
        canonical_ps = members[medoid_idx].problem_statement

        # spans
        drivers = {m.experience_driver for m in members}
        streams = {m.opportunity_stream for m in members}
        emotions = {m.emotion for m in members}
        activation_mix = sorted({m.activation_status for m in members})

        # convergence strength = mean of pairwise above threshold inside the cluster
        mask = sub_sims >= similarity_threshold
        if mask.sum() > 0:
            conv_strength = float(sub_sims[mask].mean())
        else:
            conv_strength = float(sub_sims.mean())

        # lead OU selection
        lead_ou = max(members, key=score_ou_for_lead)

        csli_group = {
            "csli_group_id": f"CSLI-GRP-{gid_counter:04d}",
            "generated_at": now.isoformat(timespec="seconds") + "Z",
            "version": "CSLI.v1",
            "canonical_problem_statement": canonical_ps,
            "convergence_strength_score": round(conv_strength, 3),
            "convergence_threshold_used": similarity_threshold,
            "driver_span_count": len(drivers),
            "stream_span_count": len(streams),
            "emotion_span": sorted(list(emotions)),
            "activation_mix": activation_mix,
            "member_ous": [
                {
                    "ou_id": m.ou_id,
                    "experience_driver": m.experience_driver,
                    "opportunity_stream": m.opportunity_stream,
                    "emotion": m.emotion,
                    "activation_status": m.activation_status,
                    "eri_score": m.eri_score,
                    "rf_urgency_category": m.rf_urgency_category,
                } for m in members
            ],
            "lead_ou_id": lead_ou.ou_id,
            "lead_selection_basis": "Highest ERI × RF urgency",
            # URLs are optional, constructed from your routing layer:
            "pdca_link": f"/ou/{lead_ou.ou_id}/pdca",
            "action_path": {
                "recommendation": "Execute PDCA on lead OU.",
                "propagation_instruction": "Mark other OUs as resolved via this CSLI group once systemic fix ships.",
                "strategic_impact": "Systemic fix addresses shared root cause across member OUs."
            },
            "why_csli_triggered": [
                "Multiple Experience Drivers point to identical root cause.",
                "Emotional convergence detected despite differing contexts.",
                "Prevention of duplicated effort and fragmented fixes."
            ],
            # Memory cross-refs are optional—fill if you maintain these indices:
            "ime_reference": {
                "related_outcome_evaluations": [],
                "silent_guidance_links": []
            },
            "forecast": {  # optional; remove if you prefer CSLI to be strictly descriptive
                "potential_emotional_shift": None,
                "projected_eri_lift": None,
                "projected_elasticity_gain": None,
                "systemic_risk_reduction": True
            }
        }
        gid_counter += 1
        outputs.append(csli_group)

    return outputs
```

# ### Notes (CSLI)

# * **All fields are computable** from: OU table (incl. problem statements), ERI/RF snapshots, activation status.
# * **Links** like `pdca_link` are **optional** (derive from your router + IDs).
# * **Forecast** is optional; if you want zero inference, drop that block.

# ---

# # Silent Guidance – guaranteed-computable fields

# Silent Guidance = **pattern recall** over IME. No fantasy fields; all come from IME + PDCA.

```python
# silent_guidance.py
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class MissionSummary:
    ou_id: str
    experience_driver: str
    emotion: str
    opportunity_stream: str
    interaction_moment: Optional[str]
    initiatives: List[str]
    outcomes: List[str]
    impact_kpis: Dict[str, float]  # whatever you already log (e.g., complaint_rate_delta)

class IME:
    def search_past_missions(self,
        experience_driver: Optional[str],
        emotion_group: Optional[str],
        opportunity_stream: Optional[str],
        interaction_moment: Optional[str],
        limit:int=10) -> List[MissionSummary]:
        raise NotImplementedError

def silent_guidance(
    ime: IME,
    seed: Dict[str, Optional[str]],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    seed: {
      "experience_driver": str,
      "emotion_group": str,
      "opportunity_stream": str,
      "interaction_moment": Optional[str]
    }
    """
    now = now or datetime.utcnow()
    matches = ime.search_past_missions(
        experience_driver=seed.get("experience_driver"),
        emotion_group=seed.get("emotion_group"),
        opportunity_stream=seed.get("opportunity_stream"),
        interaction_moment=seed.get("interaction_moment"),
        limit=10
    )

    guidance = {
        "silent_guidance_id": f"SG-{int(now.timestamp())}",
        "generated_at": now.isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "matches": [
            {
                "ou_id": m.ou_id,
                "experience_driver": m.experience_driver,
                "emotion": m.emotion,
                "opportunity_stream": m.opportunity_stream,
                "interaction_moment": m.interaction_moment,
                "initiatives": m.initiatives,
                "outcomes": m.outcomes,
                "impact_kpis": m.impact_kpis,
                "pdca_link": f"/ou/{m.ou_id}/pdca"
            } for m in matches
        ],
        "pattern_summary": {
            "count": len(matches),
            "owner_history": {},          # fill from your IME if you track owners
            "common_initiatives": {},     # simple frequency aggregation
            "feasibility_patterns": {}    # aggregate from prior CHECK phase scores if stored
        },
        "recommendation": "Reuse high-ROI initiatives from top-matching missions; avoid low-feasibility patterns."
    }
    return guidance
```

# **Everything here is retrievable** from IME snapshots of past PDCA missions. No invented fields.

# ---

# SCE (Silent Cognition Engine) – guaranteed-computable fields

SCE runs on **EDs with activated OUs** and checks **post-activation ERI/RF series**. All fields below come from telemetry + IME.

```python
# sce.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

@dataclass
class EDActivation:
    experience_driver: str
    last_ou_id: str
    activated_at: datetime

class Telemetry:
    def fetch_eri_rf_series(self, experience_driver: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        returns df with index=timestamp, columns=['eri','rf']
        """
        raise NotImplementedError

class IME:
    def list_last_activations(self) -> List[EDActivation]:
        raise NotImplementedError
    def get_ou_header(self, ou_id: str) -> Dict[str, Any]:
        raise NotImplementedError

def classify_silence(df: pd.DataFrame, min_events:int=1) -> bool:
    return df.shape[0] < min_events  # no signal points in the window

def sce_silence_scan(
    ime: IME,
    telemetry: Telemetry,
    window_days: int = 90,
    now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    now = now or datetime.utcnow()
    results = []
    activations = ime.list_last_activations()

    for act in activations:
        start = act.activated_at
        end = now
        df = telemetry.fetch_eri_rf_series(act.experience_driver, start, end)

        silent = classify_silence(df)
        silence_duration_days = (end - start).days if silent else 0

        # basic heuristics (all computable):
        eri_before = ime.get_ou_header(act.last_ou_id).get("eri_before")
        eri_after = ime.get_ou_header(act.last_ou_id).get("eri_after")
        stability = "Stable" if not df.empty and df['eri'].std() < 5 else "Unstable"

        if silent and (eri_after is not None) and (eri_after > (eri_before or -999)):
            diagnosis = "Resolution"
            action = "Suggest Amplify"
        elif silent:
            diagnosis = "Disengagement Risk"
            action = "Flag for Revisit"
        else:
            diagnosis = "Active Signal"
            action = "Monitor"

        results.append({
            "silence_analysis_id": f"SCE-{act.experience_driver.replace(' ','_')}-{int(now.timestamp())}",
            "generated_at": now.isoformat(timespec="seconds") + "Z",
            "detection_summary": {
                "experience_driver": act.experience_driver,
                "silence_detected": bool(silent),
                "silence_duration_days": silence_duration_days,
                "silence_scan_window": f"{window_days}-day post-activation",
                "last_known_ou_id": act.last_ou_id
            },
            "ou_activation_history": {
                "ou_summary": {
                    "ou_id": act.last_ou_id
                },
                "emotion_diagnostics": {
                    "eri_before": eri_before,
                    "eri_after": eri_after,
                    "post_series_stability": stability
                }
            },
            "heuristic_classification": {
                "diagnosis": diagnosis,
                "suggested_action": action
            }
        })
    return results
```

# **Everything here is measurable**: ERI/RF time series, OU activation timestamps, pre/post ERI from Outcome Evaluation or PDCA memory headers.

# ---

# ## TL;DR assurances

# * **CSLI**: computed from OU table (problem statements + metadata) + embeddings. No fantasy fields.
# * **Silent Guidance**: computed from IME past missions. Pure lookup + aggregation.
# * **SCE**: computed from IME activations + ERI/RF telemetry within a time window.

# If you want, I can bundle these into a tiny **package layout** (`/ime`, `/csli`, `/sce`) with interfaces and a couple of unit-test stubs so your 
# engineers can wire in your real storage/telemetry. Want me to draft that folder structure next?
