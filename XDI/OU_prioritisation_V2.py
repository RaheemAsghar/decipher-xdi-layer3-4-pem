# ou_temporal_filter.py

from typing import Any, Dict, List, Optional


class TemporalIntelligenceFilter:
    """
    OU temporal filter that uses ONLY:
      - Emotion_Recency_Profile.by_emotion
      - Emotion_Recency_Profile.by_tier
      - Emotion_Recency_Profile.temporal_intelligence
    computed in Layer_2_V7.

    NO NEW METRICS. PURE DERIVATION.

    Responsibilities:
    - Decide which OUs to ACTIVATE vs DEFER based on:
        • Emotion recency concentration (recent vs early tiers)
        • Tier-level presence/absence
        • Emotion momentum (rising / fading / stable)
        • Dominant emotion transition (early → recent)
        • Pattern label context (crisis / recovery)
    - Provide a confidence score (0–100) and rationale.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Optional config to tune thresholds. All are applied on top of
        Layer_2_V7 outputs without changing the underlying math.
        """
        defaults: Dict[str, Any] = {
            "recency_high": 60.0,      # % of emotion in Tier_1+2 to call it "highly recent"
            "recency_moderate": 40.0,  # % to call it "moderately recent"
            "recency_low": 20.0,       # below this → likely historical/fading
            "tier_absence": 3,         # how many of Tier_1/2/3 can be empty before treating as "absent"
            "momentum_strong": 15.0,   # Δ pct points to treat change as strongly rising/fading
            "momentum_moderate": 8.0,  # Δ pct points to treat change as moderately rising/fading
        }

        cfg = config or {}

        self.RECENCY_THRESHOLD_HIGH = float(
            cfg.get("recency_high", defaults["recency_high"])
        )
        self.RECENCY_THRESHOLD_MODERATE = float(
            cfg.get("recency_moderate", defaults["recency_moderate"])
        )
        self.RECENCY_THRESHOLD_LOW = float(
            cfg.get("recency_low", defaults["recency_low"])
        )
        self.TIER_ABSENCE_THRESHOLD = int(
            cfg.get("tier_absence", defaults["tier_absence"])
        )
        self.MOMENTUM_STRONG_THRESHOLD = float(
            cfg.get("momentum_strong", defaults["momentum_strong"])
        )
        self.MOMENTUM_MODERATE_THRESHOLD = float(
            cfg.get("momentum_moderate", defaults["momentum_moderate"])
        )

    # =====================================================================
    # PUBLIC ENTRYPOINT
    # =====================================================================

    def filter_ous_by_temporal_intelligence(
        self,
        ou_candidates: List[Dict[str, Any]],
        seu_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Filter OU candidates using SEU's Recency Profile + Temporal Intelligence.

        Args:
            ou_candidates: List of OU dicts with at least:
                - ou_id
                - ou_name
                - dominant_emotion (or primary_emotion / emotion)
                - mention_count (not used in logic but often available)
            seu_row: Full SEU row with Emotion_Recency_Profile field.

        Returns:
            {
                "activate": [ {..ou.., "temporal_decision": {...}} ],
                "defer":    [ {..ou.., "temporal_decision": {...}} ],
                "insights": {
                    "filter_applied": bool,
                    "total_ous": int,
                    "activated": int,
                    "deferred": int,
                    "activation_rate_pct": float,
                    "top_defer_reason": str,
                    "temporal_context": {...},
                    "summary": str,
                }
            }
        """
        recency_profile: Dict[str, Any] = seu_row.get("Emotion_Recency_Profile", {}) or {}
        temporal_intel: Dict[str, Any] = recency_profile.get("temporal_intelligence", {}) or {}

        # If Layer_2_V7 did not produce temporal intelligence, do NOT invent it.
        if not recency_profile or not temporal_intel:
            return {
                "activate": ou_candidates,
                "defer": [],
                "insights": {
                    "filter_applied": False,
                    "reason": "No temporal intelligence available in SEU",
                },
            }

        activate_list: List[Dict[str, Any]] = []
        defer_list: List[Dict[str, Any]] = []

        for ou in ou_candidates:
            decision = self._evaluate_ou(ou, recency_profile, temporal_intel)

            enriched = {**ou, "temporal_decision": decision}
            if decision["action"] == "ACTIVATE":
                activate_list.append(enriched)
            else:
                defer_list.append(enriched)

        insights = self._generate_insights(activate_list, defer_list, temporal_intel)

        return {
            "activate": activate_list,
            "defer": defer_list,
            "insights": insights,
        }

    # =====================================================================
    # CORE EVALUATION LOGIC
    # =====================================================================

    def _evaluate_ou(
        self,
        ou: Dict[str, Any],
        recency_profile: Dict[str, Any],
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single OU using ONLY Layer_2_V7 temporal constructs.

        Returns:
            {
                "action": "ACTIVATE" | "DEFER",
                "confidence": int 0–100,
                "rationale": [str, ...],
                "signals": {
                    "recency": {...},
                    "absence": {...},
                    "momentum": {...},
                    "transition": {...},
                    "pattern": {...},
                }
            }
        """
        rationale: List[str] = []
        confidence: float = 50.0  # neutral baseline

        # Extract OU's dominant emotion (several possible keys)
        dominant_emotion: Optional[str] = (
            ou.get("dominant_emotion")
            or ou.get("primary_emotion")
            or ou.get("emotion")
        )

        if not dominant_emotion:
            # Cannot apply temporal logic if we don't know the emotion.
            return {
                "action": "ACTIVATE",  # never block if we genuinely don't know
                "confidence": 50,
                "rationale": [
                    "No emotion tag available – defaulting to ACTIVATE without temporal filter."
                ],
            }

        # Normalize to match Layer_2_V7 emotion keys: "Anger", "Agitation", etc.
        dominant_emotion = str(dominant_emotion).strip().title()

        # =====================================================================
        # CHECK 1: EMOTION RECENCY (by_emotion)
        # =====================================================================
        recency_signal = self._check_emotion_recency(dominant_emotion, recency_profile)

        if recency_signal["available"]:
            if recency_signal["recent_pct"] >= self.RECENCY_THRESHOLD_HIGH:
                confidence += 30
                rationale.append(
                    f"✅ {dominant_emotion} is HIGHLY RECENT "
                    f"({recency_signal['recent_pct']:.1f}% in Tier_1+2)."
                )
            elif recency_signal["recent_pct"] >= self.RECENCY_THRESHOLD_MODERATE:
                confidence += 15
                rationale.append(
                    f"✅ {dominant_emotion} is MODERATELY RECENT "
                    f"({recency_signal['recent_pct']:.1f}% in Tier_1+2)."
                )
            elif recency_signal["recent_pct"] < self.RECENCY_THRESHOLD_LOW:
                confidence -= 30
                rationale.append(
                    f"❌ {dominant_emotion} is HISTORICAL/FADING "
                    f"({recency_signal['recent_pct']:.1f}% in Tier_1+2, "
                    f"{recency_signal['early_pct']:.1f}% in Tier_4+5)."
                )
        else:
            rationale.append(
                f"ℹ️ No reliable recency distribution for {dominant_emotion} in by_emotion – "
                f"no confidence adjustment based on recency tiers."
            )

        # =====================================================================
        # CHECK 2: TIER-LEVEL ABSENCE / PRESENCE (by_tier)
        # =====================================================================
        absence_signal = self._check_tier_absence(dominant_emotion, recency_profile)

        if absence_signal["absent_from_top_n"] >= self.TIER_ABSENCE_THRESHOLD:
            confidence -= 40
            rationale.append(
                f"❌ {dominant_emotion} ABSENT from {absence_signal['absent_from_top_n']} "
                f"of top tiers {absence_signal['checked_tiers']} – likely RESOLVED/HISTORICAL."
            )
        elif absence_signal["present_in_tier_1"]:
            confidence += 10
            rationale.append(
                f"✅ {dominant_emotion} PRESENT in Tier_1 – strongly current in latest window."
            )

        # =====================================================================
        # CHECK 3: EMOTION MOMENTUM (Emotion_Flow.shifts)
        # =====================================================================
        momentum_signal = self._check_emotion_momentum(dominant_emotion, temporal_intel)

        if momentum_signal["available"]:
            trend = momentum_signal["trend"]
            delta = momentum_signal["delta"]

            if trend in ("↑↑ Strongly Emerging", "↑ Rising"):
                # rising / emerging
                if abs(delta) >= self.MOMENTUM_STRONG_THRESHOLD:
                    confidence += 25
                elif abs(delta) >= self.MOMENTUM_MODERATE_THRESHOLD:
                    confidence += 15
                rationale.append(
                    f"✅ {dominant_emotion} is {trend} "
                    f"(Δ={delta:+.1f} pct points)."
                )
            elif trend in ("↓↓ Rapidly Fading", "↓ Declining"):
                # fading / declining
                if abs(delta) >= self.MOMENTUM_STRONG_THRESHOLD:
                    confidence -= 25
                elif abs(delta) >= self.MOMENTUM_MODERATE_THRESHOLD:
                    confidence -= 15
                rationale.append(
                    f"❌ {dominant_emotion} is {trend} "
                    f"(Δ={delta:+.1f} pct points)."
                )
            elif trend == "→ Stable":
                rationale.append(
                    f"➡️ {dominant_emotion} momentum is STABLE "
                    f"(Δ={delta:+.1f} pct points)."
                )
        else:
            # No momentum info for this emotion – do nothing, just log.
            rationale.append(
                f"ℹ️ No emotion momentum data available for {dominant_emotion} in Emotion_Flow."
            )

        # =====================================================================
        # CHECK 4: DOMINANT EMOTION TRANSITION (Emotion_Flow.transition)
        # =====================================================================
        transition_signal = self._check_dominant_transition(dominant_emotion, temporal_intel)

        if transition_signal["available"]:
            if transition_signal["is_new_dominant"]:
                confidence += 20
                rationale.append(
                    f"✅ {dominant_emotion} is the NEW dominant emotion "
                    f"({transition_signal['transition']})."
                )
            elif transition_signal["is_old_dominant"]:
                confidence -= 20
                rationale.append(
                    f"❌ {dominant_emotion} is the OLD dominant emotion "
                    f"({transition_signal['transition']})."
                )
            else:
                rationale.append(
                    f"➡️ {dominant_emotion} is not the main driver of the current "
                    f"dominant transition ({transition_signal['transition']})."
                )
        else:
            rationale.append(
                "ℹ️ No dominant emotion transition info available – "
                "treating dominant flow as stable or unspecified."
            )

        # =====================================================================
        # CHECK 5: PATTERN CONTEXT (Pattern_Flow.pattern_label)
        # =====================================================================
        pattern_signal = self._check_pattern_context(dominant_emotion, temporal_intel)

        if pattern_signal["is_crisis_pattern"] and dominant_emotion in {"Anger", "Agitation"}:
            confidence += 20
            rationale.append(
                f"✅ Pattern '{pattern_signal['pattern_label']}' indicates CRISIS and "
                f"supports focusing on {dominant_emotion}."
            )
        elif pattern_signal["is_recovery_pattern"] and dominant_emotion in {"Adoration", "Appreciation"}:
            confidence += 20
            rationale.append(
                f"✅ Pattern '{pattern_signal['pattern_label']}' indicates RECOVERY and "
                f"supports focusing on {dominant_emotion}."
            )
        elif pattern_signal["pattern_label"] != "Unknown":
            rationale.append(
                f"ℹ️ Pattern '{pattern_signal['pattern_label']}' does not strongly favour "
                f"{dominant_emotion} one way or the other."
            )

        # =====================================================================
        # FINAL DECISION
        # =====================================================================
        # Clamp confidence to [0, 100]
        confidence_int = int(max(0, min(100, round(confidence))))

        # Threshold: 60+ → ACTIVATE, else DEFER
        action = "ACTIVATE" if confidence_int >= 60 else "DEFER"

        return {
            "action": action,
            "confidence": confidence_int,
            "rationale": rationale,
            "signals": {
                "recency": recency_signal,
                "absence": absence_signal,
                "momentum": momentum_signal,
                "transition": transition_signal,
                "pattern": pattern_signal,
            },
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
                "is_old_dominant": False,
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
        Query: Does the overall pattern suggest this emotion is relevant?

        Uses:
            temporal_intel["Pattern_Flow"]["pattern_label"]
        and only interprets it (no relabeling).
        """
        pattern_flow = temporal_intel.get("Pattern_Flow", {}) or {}
        pattern_label = pattern_flow.get("pattern_label", "")

        if not pattern_label:
            return {
                "pattern_label": "Unknown",
                "pattern_supports": False,
                "is_crisis_pattern": False,
                "is_recovery_pattern": False,
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

        pattern_supports = False
        if emotion in {"Anger", "Agitation"} and is_crisis:
            pattern_supports = True
        if emotion in {"Adoration", "Appreciation"} and is_recovery:
            pattern_supports = True

        return {
            "pattern_label": pattern_label,
            "pattern_supports": pattern_supports,
            "is_crisis_pattern": is_crisis,
            "is_recovery_pattern": is_recovery,
        }

    # =====================================================================
    # INSIGHTS
    # =====================================================================

    def _generate_insights(
        self,
        activate_list: List[Dict[str, Any]],
        defer_list: List[Dict[str, Any]],
        temporal_intel: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate human-readable summary of filtering decisions.

        All context fields are taken directly from Layer_2_V7 temporal_intelligence.
        """
        total = len(activate_list) + len(defer_list)

        if total == 0:
            return {
                "filter_applied": True,
                "total_ous": 0,
                "activated": 0,
                "deferred": 0,
                "activation_rate_pct": 0.0,
                "top_defer_reason": "N/A",
                "temporal_context": {},
                "summary": "No OUs provided to temporal filter.",
            }

        activated_pct = (len(activate_list) / total) * 100.0

        # Aggregate defer reasons
        defer_reasons: List[str] = []
        for ou in defer_list:
            td = ou.get("temporal_decision", {})
            for r in td.get("rationale", []):
                if r.startswith("❌"):
                    defer_reasons.append(r)

        from collections import Counter

        reason_counts = Counter(defer_reasons)
        top_defer_reason = reason_counts.most_common(1)[0][0] if reason_counts else "N/A"

        eri_flow = temporal_intel.get("ERI_Flow", {}) or {}
        pattern_flow = temporal_intel.get("Pattern_Flow", {}) or {}

        summary = (
            f"Temporal filter activated {len(activate_list)}/{total} OUs "
            f"({activated_pct:.0f}%). "
            f"Deferred {len(defer_list)} OUs primarily when signals were "
            f"historical/fading or resolved. "
            f"Overall pattern: {pattern_flow.get('pattern_label', 'Unknown')}. "
            f"ERI trend: {eri_flow.get('trend', 'Unknown')}."
        )

        return {
            "filter_applied": True,
            "total_ous": total,
            "activated": len(activate_list),
            "deferred": len(defer_list),
            "activation_rate_pct": round(activated_pct, 1),
            "top_defer_reason": top_defer_reason,
            "temporal_context": {
                "eri_trend": eri_flow.get("trend"),
                "pattern": pattern_flow.get("pattern_label"),
            },
            "summary": summary,
        }
