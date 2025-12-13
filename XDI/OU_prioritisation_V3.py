# ou_temporal_enrichment.py

from typing import Any, Dict, List, Optional


class TemporalIntelligenceEnricher:
    """
    OU temporal enrichment that uses ONLY:
      - Emotion_Recency_Profile.by_emotion
      - Emotion_Recency_Profile.by_tier
      - Emotion_Recency_Profile.temporal_intelligence
    computed in Layer_2_V7.

    NO NEW METRICS. PURE DERIVATION.

    Responsibilities:
    - Enrich each OU with temporal signals from SEU's recency profile
    - Provide contextual information about:
        • Emotion recency concentration (recent vs early tiers)
        • Tier-level presence/absence
        • Emotion momentum (rising / fading / stable)
        • Dominant emotion transition (early → recent)
        • Pattern label context (crisis / recovery)
    - No decisions, no filtering - just enrichment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Minimal initialization - no thresholds needed since we're just enriching.
        """
        pass

    # =====================================================================
    # PUBLIC ENTRYPOINT
    # =====================================================================

    def enrich_ous_with_temporal_intelligence(
        self,
        ou_candidates: List[Dict[str, Any]],
        seu_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich OU candidates with SEU's Recency Profile + Temporal Intelligence.

        Args:
            ou_candidates: List of OU dicts with at least:
                - ou_id
                - ou_name
                - dominant_emotion (or primary_emotion / emotion)
                - mention_count (not used in logic but often available)
            seu_row: Full SEU row with Emotion_Recency_Profile field.

        Returns:
            {
                "enriched_ous": [ {..ou.., "temporal_signals": {...}} ],
                "insights": {
                    "enrichment_applied": bool,
                    "total_ous": int,
                    "temporal_context": {...},
                    "summary": str,
                }
            }
        """
        recency_profile: Dict[str, Any] = seu_row.get("Emotion_Recency_Profile", {}) or {}
        temporal_intel: Dict[str, Any] = recency_profile.get("temporal_intelligence", {}) or {}

        # If Layer_2_V7 did not produce temporal intelligence, return without enrichment
        if not recency_profile or not temporal_intel:
            return {
                "enriched_ous": ou_candidates,
                "insights": {
                    "enrichment_applied": False,
                    "reason": "No temporal intelligence available in SEU",
                },
            }

        enriched_list: List[Dict[str, Any]] = []

        for ou in ou_candidates:
            signals = self._extract_temporal_signals(ou, recency_profile, temporal_intel)
            enriched = {**ou, "temporal_signals": signals}
            enriched_list.append(enriched)

        insights = self._generate_insights(enriched_list, temporal_intel)

        return {
            "enriched_ous": enriched_list,
            "insights": insights,
        }

    # =====================================================================
    # CORE ENRICHMENT LOGIC
    # =====================================================================

    def _extract_temporal_signals(
        self,
        ou: Dict[str, Any],
        recency_profile: Dict[str, Any],
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract temporal signals for a single OU using ONLY Layer_2_V7 temporal constructs.

        Returns:
            {
                "recency": {...},
                "absence": {...},
                "momentum": {...},
                "transition": {...},
                "pattern": {...},
            }
        """
        # Extract OU's dominant emotion (several possible keys)
        dominant_emotion: Optional[str] = (
            ou.get("dominant_emotion")
            or ou.get("primary_emotion")
            or ou.get("emotion")
        )

        if not dominant_emotion:
            # Cannot apply temporal logic if we don't know the emotion
            return {
                "recency": {"available": False},
                "absence": {"available": False},
                "momentum": {"available": False},
                "transition": {"available": False},
                "pattern": {"available": False},
                "note": "No emotion tag available for temporal enrichment"
            }

        # Normalize to match Layer_2_V7 emotion keys: "Anger", "Agitation", etc.
        dominant_emotion = str(dominant_emotion).strip().title()

        # =====================================================================
        # EXTRACT ALL TEMPORAL SIGNALS
        # =====================================================================
        recency_signal = self._check_emotion_recency(dominant_emotion, recency_profile)
        absence_signal = self._check_tier_absence(dominant_emotion, recency_profile)
        momentum_signal = self._check_emotion_momentum(dominant_emotion, temporal_intel)
        transition_signal = self._check_dominant_transition(dominant_emotion, temporal_intel)
        pattern_signal = self._check_pattern_context(dominant_emotion, temporal_intel)

        return {
            "recency": recency_signal,
            "absence": absence_signal,
            "momentum": momentum_signal,
            "transition": transition_signal,
            "pattern": pattern_signal,
        }

    # =====================================================================
    # LOW-LEVEL HELPERS – PURE QUERIES OVER LAYER_2_V7 STRUCTURES
    # =====================================================================

    def _check_emotion_recency(
        self,
        emotion: str,
        recency_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query: What % of this emotion's mentions are in recent vs early tiers?

        Uses:
            Emotion_Recency_Profile["by_emotion"] exactly as Layer_2_V7 created it.
        """
        by_emotion = recency_profile.get("by_emotion", {}) or {}

        if emotion not in by_emotion:
            return {
                "recent_pct": 0.0,
                "early_pct": 0.0,
                "recent_count": 0,
                "early_count": 0,
                "total_count": 0,
                "available": False,
            }

        emotion_data = by_emotion.get(emotion, {})
        tiers = emotion_data.get("tiers", {}) or {}

        # Recent = Tier_1 + Tier_2
        recent_count = (
            tiers.get("Tier_1", {}).get("count", 0)
            + tiers.get("Tier_2", {}).get("count", 0)
        )

        # Early = Tier_4 + Tier_5
        early_count = (
            tiers.get("Tier_4", {}).get("count", 0)
            + tiers.get("Tier_5", {}).get("count", 0)
        )

        total_count = int(emotion_data.get("total_count", 0))

        if total_count <= 0:
            return {
                "recent_pct": 0.0,
                "early_pct": 0.0,
                "recent_count": recent_count,
                "early_count": early_count,
                "total_count": total_count,
                "available": False,
            }

        recent_pct = (recent_count / total_count) * 100.0
        early_pct = (early_count / total_count) * 100.0

        return {
            "recent_pct": round(recent_pct, 1),
            "early_pct": round(early_pct, 1),
            "recent_count": recent_count,
            "early_count": early_count,
            "total_count": total_count,
            "available": True,
        }

    def _check_tier_absence(
        self,
        emotion: str,
        recency_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query: Is this emotion absent from recent tiers?

        Uses:
            Emotion_Recency_Profile["by_tier"]
        """
        by_tier = recency_profile.get("by_tier", {}) or {}
        recent_tiers = ["Tier_1", "Tier_2", "Tier_3"]

        absent_count = 0
        present_in_tier_1 = False

        for tier_name in recent_tiers:
            tier_data = by_tier.get(tier_name, {}) or {}
            emotions_in_tier = tier_data.get("emotions", {}) or {}

            if emotion not in emotions_in_tier or emotions_in_tier.get(emotion, {}).get("count", 0) == 0:
                absent_count += 1

            if tier_name == "Tier_1" and emotion in emotions_in_tier:
                if emotions_in_tier[emotion].get("count", 0) > 0:
                    present_in_tier_1 = True

        return {
            "absent_from_top_n": absent_count,
            "present_in_tier_1": present_in_tier_1,
            "checked_tiers": recent_tiers,
            "available": True,
        }

    def _check_emotion_momentum(
        self,
        emotion: str,
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query: Is this emotion rising, declining, or stable?

        Uses:
            temporal_intel["Emotion_Flow"]["shifts"]
        as produced by Layer_2_V7.
        """
        emotion_flow = temporal_intel.get("Emotion_Flow", {}) or {}
        shifts = emotion_flow.get("shifts", {}) or {}

        if emotion not in shifts:
            return {
                "trend": "Unknown",
                "delta": 0.0,
                "old_pct": 0.0,
                "new_pct": 0.0,
                "available": False,
            }

        shift_data = shifts.get(emotion, {})

        return {
            "trend": shift_data.get("trend", "Unknown"),
            "delta": shift_data.get("delta", 0.0),
            "old_pct": shift_data.get("old_pct", 0.0),
            "new_pct": shift_data.get("new_pct", 0.0),
            "available": True,
        }

    def _check_dominant_transition(
        self,
        emotion: str,
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query: Is this emotion the new or old dominant emotion?

        Uses:
            temporal_intel["Emotion_Flow"]["dominant_emotion_transition"]
        as created by Layer_2_V7.
        """
        emotion_flow = temporal_intel.get("Emotion_Flow", {}) or {}
        transition = emotion_flow.get("dominant_emotion_transition", "")

        if not transition or "→" not in transition:
            # Stable or no explicit transition string
            dom_early = emotion_flow.get("dominant_emotion_early")
            dom_recent = emotion_flow.get("dominant_emotion_recent")

            return {
                "transition": transition or "Stable",
                "is_new_dominant": emotion == dom_recent,
                "is_old_dominant": emotion == dom_early,
                "available": bool(dom_recent),
            }

        # Parse "Agitation → Anger"
        parts = transition.split(" → ")
        old_emotion = parts[0].strip() if len(parts) > 0 else ""
        new_emotion = parts[1].strip() if len(parts) > 1 else ""

        return {
            "transition": transition,
            "is_new_dominant": emotion == new_emotion,
            "is_old_dominant": emotion == old_emotion,
            "available": True,
        }

    def _check_pattern_context(
        self,
        emotion: str,
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query: What is the overall detected pattern context (label + crisis/recovery flags)?

        Uses:
            temporal_intel["Pattern_Flow"]["pattern_label"]
        and only interprets it (no relabeling).
        """
        pattern_flow = temporal_intel.get("Pattern_Flow", {}) or {}
        pattern_label = pattern_flow.get("pattern_label", "")

        if not pattern_label:
            return {
                "pattern_label": "Unknown",
                "is_crisis_pattern": False,
                "is_recovery_pattern": False,
                "available": False,
            }

        pattern_upper = pattern_label.upper()

        crisis_prefixes = (
            "CRISIS",
            "CATASTROPH",
            "MELTDOWN",
            "SEVERE DETERIORATION",
        )
        recovery_prefixes = (
            "RECOVERY",
            "IMPROV",
            "HEAL",
            "POSITIVE SHIFT",
        )

        is_crisis = any(pattern_upper.startswith(p) for p in crisis_prefixes)
        is_recovery = any(pattern_upper.startswith(p) for p in recovery_prefixes)

        return {
            "pattern_label": pattern_label,
            "is_crisis_pattern": is_crisis,
            "is_recovery_pattern": is_recovery,
            "available": True,
        }

    # =====================================================================
    # INSIGHTS
    # =====================================================================

    def _generate_insights(
        self,
        enriched_list: List[Dict[str, Any]],
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate human-readable summary of enrichment process.

        All context fields are taken directly from Layer_2_V7 temporal_intelligence.
        """
        total = len(enriched_list)

        if total == 0:
            return {
                "enrichment_applied": True,
                "total_ous": 0,
                "temporal_context": {},
                "summary": "No OUs provided for temporal enrichment.",
            }

        eri_flow = temporal_intel.get("ERI_Flow", {}) or {}
        pattern_flow = temporal_intel.get("Pattern_Flow", {}) or {}

        summary = (
            f"Temporal enrichment applied to {total} OUs. "
            f"Overall pattern: {pattern_flow.get('pattern_label', 'Unknown')}. "
            f"ERI trend: {eri_flow.get('trend', 'Unknown')}."
        )

        return {
            "enrichment_applied": True,
            "total_ous": total,
            "temporal_context": {
                "eri_trend": eri_flow.get("trend"),
                "pattern": pattern_flow.get("pattern_label"),
            },
            "summary": summary,
        }

