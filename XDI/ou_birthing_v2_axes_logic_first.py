"""OU Birthing — v2 Axes (Logic-First)

Aligns OU construction to the new schema:

- intent_axis: string
- affect_axis: { affect_label: string, confidence_level: string }
- action_axis: string
- journey: { patient_journey, journey_stage, interaction_moment }
- context: { text, keywords }
- orchestration_unit: { matters, semantic_action_statement{...}, ... }

Recommended OU identity
----------------------
experience_driver (canonical) +
intent_axis +
action_axis +
affect_label +
behavioral cluster signature (deterministic hash)

This file provides:
- a v2 signature builder (replaces old emotion/stream fields)
- a stable OU fingerprint helper (Memory Law friendly)
- strict v2 OU payload shaping
- a minimal single-mention OU birth primitive (your full pipeline will cluster many mentions)

It is intentionally logic-first and dependency-light.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib


DEFAULT_SIGNATURE_WEIGHTS: Dict[str, float] = {
    "intent_axis": 2.0,
    "action_axis": 2.0,
    "affect_label": 1.5,
    "interaction_moment": 1.5,
    "context_text": 2.0,
    "matters": 2.5,
    "sas_section_1_patient_reality": 2.0,
    "sas_section_2_strategic_response": 2.0,
    "keywords": 1.0,
}


def _safe(s: Optional[str]) -> str:
    return (s or "").strip()


def build_signature(mention: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> str:
    w = weights or DEFAULT_SIGNATURE_WEIGHTS
    parts: List[str] = []

    def add(key: str, value: str):
        value = _safe(value)
        if not value:
            return
        reps = max(1, int(round(w.get(key, 1.0))))
        parts.extend([f"{key}: {value}"] * reps)

    add("intent_axis", mention.get("intent_axis", ""))
    add("action_axis", mention.get("action_axis", ""))

    affect = mention.get("affect_axis") or {}
    add("affect_label", affect.get("affect_label", ""))

    journey = mention.get("journey") or {}
    add("interaction_moment", journey.get("interaction_moment", ""))

    ctx = mention.get("context") or {}
    add("context_text", ctx.get("text", ""))

    ou = mention.get("orchestration_unit") or {}
    add("matters", ou.get("matters", ""))

    sas = ou.get("semantic_action_statement") or {}
    add("sas_section_1_patient_reality", sas.get("section_1_patient_reality", ""))
    add("sas_section_2_strategic_response", sas.get("section_2_strategic_response", ""))

    kws = ctx.get("keywords") or []
    if isinstance(kws, list) and kws:
        add("keywords", ", ".join([str(x) for x in kws[:12]]))

    return " | ".join(parts)


def stable_ou_id(
    experience_driver_label: str,
    intent_axis: str,
    action_axis: str,
    affect_label: str,
    signature_repr: str,
    *,
    salt: str = "ou_v2_axes"
) -> str:
    base = "||".join([
        salt,
        _safe(experience_driver_label).lower(),
        _safe(intent_axis).lower(),
        _safe(action_axis).lower(),
        _safe(affect_label).lower(),
        _safe(signature_repr).lower(),
    ])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:14]


def build_ou_payload_v2(mention: Dict[str, Any], *, ou_id: str) -> Dict[str, Any]:
    ed = mention.get("experience_driver") or {}
    ou = mention.get("orchestration_unit") or {}
    affect = mention.get("affect_axis") or {}

    return {
        "ou_id": ou_id,
        "ou_name": ou.get("ou_name", ""),

        "experience_driver": ed.get("label", ou.get("experience_driver", "")),
        "entity_name": mention.get("entity_name", ou.get("entity_name", "")),
        "theme": mention.get("theme", ou.get("theme", "")),

        "intent_axis": mention.get("intent_axis", ou.get("intent_axis", "")),
        "action_axis": mention.get("action_axis", ou.get("action_axis", "")),
        "affect_axis": {
            "affect_label": affect.get("affect_label", (ou.get("affect_axis") or {}).get("affect_label", "")),
            "confidence_level": affect.get("confidence_level", (ou.get("affect_axis") or {}).get("confidence_level", "")),
        },

        "interaction_moment": (mention.get("journey") or {}).get("interaction_moment", ou.get("interaction_moment", "")),
        "context": (mention.get("context") or {}).get("text", ou.get("context", "")),
        "matters": ou.get("matters", ""),

        "semantic_action_statement": ou.get("semantic_action_statement", {
            "section_1_patient_reality": "",
            "section_2_strategic_response": ""
        }),

        "action_axis_justification": ou.get("action_axis_justification", ""),
        "matters_extraction_source": ou.get("matters_extraction_source", ""),
        "behavioral_impact": ou.get("behavioral_impact", ""),
    }


def birth_ou_from_single_mention(mention: Dict[str, Any]) -> Dict[str, Any]:
    ed_label = ((mention.get("experience_driver") or {}).get("label") or "").strip()
    intent_axis = (mention.get("intent_axis") or "").strip()
    action_axis = (mention.get("action_axis") or "").strip()
    affect_label = ((mention.get("affect_axis") or {}).get("affect_label") or "").strip()

    sig = build_signature(mention)
    ou_id = stable_ou_id(ed_label, intent_axis, action_axis, affect_label, sig)

    return build_ou_payload_v2(mention, ou_id=ou_id)
