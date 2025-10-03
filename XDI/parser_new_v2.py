import re
import json
import random
from datetime import datetime, timedelta
from typing import List, Optional, Union
import pandas as pd
import unicodedata


class ReviewParser:
    """Surgically accurate parser for multi-driver CX markdown blocks."""

    def __init__(self, input_path: str, max_days: int = 75):
        self.csv_path: str = input_path
        self.output_df: Optional[pd.DataFrame] = None
        self.base_time: datetime = datetime.now()
        self.max_days: int = max_days

    def _read_and_normalize(self) -> str:
        with open(self.csv_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        return content.replace('""', '"').replace('\\"', '"')

    def _extract_inline(self, label: str, block: str, *, as_int: bool = False) -> Union[str, int, None]:
        pattern = rf'- \*\*{re.escape(label)}\*\*: "(.+?)"'
        match = re.search(pattern, block)
        if match:
            val = match.group(1).strip()
            if isinstance(val, str) and val.lower() == "null":
                return None
            return int(val) if as_int else val
        return None if as_int else ""

    def _extract_block(self, label: str, block: str) -> str:
        pattern = rf'- \*\*{re.escape(label)}\*\*:\s*(.+?)(?=\n- \*\*|---|###|\Z)'
        match = re.search(pattern, block, re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = re.sub(r'["\n\r]+$', '', content)
            content = re.sub(r'^["\n\r]+', '', content)
            return content.strip()
        return ""

    def _extract_flexible(self, label: str, block: str) -> str:
        """Case-insensitive extractor that tolerates quoted/unquoted values."""
        pattern = rf'- \*\*{label}\*\*:\s*(")?(.+?)\1?(?=\n- \*\*|---|###|\Z)'
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(2).strip()
        return ""

    def _extract_keywords(self, block: str) -> List[str]:
        match = re.search(r'- \*\*Keywords\*\*: \[(.*?)\]', block)
        if not match:
            return []
        raw_keywords = match.group(1)
        try:
            return json.loads(f"[{raw_keywords}]")
        except json.JSONDecodeError:
            return [k.strip().strip('"\'') for k in raw_keywords.split(',')]

    def _split_emotion(self, emotion: str) -> tuple:
        if not emotion:
            return "", ""
        parts = re.split(r"\s*[-–—]\s*", emotion.strip(), maxsplit=1)
        core = parts[0].strip()
        specific = parts[1].strip() if len(parts) > 1 else ""
        return core, specific

    def _split_semantic_action(self, sas: str) -> tuple[str, str]:
        """
        Returns (customer_reality, strategic_response) without headings.
        Robust to case/newline variations. Falls back gracefully.
        """
        if not sas:
            return "", ""
        s = re.sub(r"\r\n?", "\n", str(sas))

        m1 = re.search(r"section\s*1\b.*?(customer\s*reality)", s, flags=re.I | re.S)
        m2 = re.search(r"\bsection\s*2\b.*?(strategic\s*response)", s, flags=re.I | re.S)

        if m1 and m2:
            start_1 = m1.end()
            start_2_hdr = m2.start()
            end_2_hdr = m2.end()
            part1 = s[start_1:start_2_hdr]
            part2 = s[end_2_hdr:]
        elif m1 and not m2:
            start_1 = m1.end()
            part1 = s[start_1:]
            part2 = ""
        elif not m1 and m2:
            start_2 = m2.end()
            part1 = s[:m2.start()]
            part2 = s[start_2:]
        else:
            split_simple = re.split(r"\bSECTION\s*2\b", s, flags=re.I)
            if len(split_simple) > 1:
                part1 = split_simple[0]
                part2 = split_simple[1]
            else:
                return s.strip(), ""

        def _clean(txt: str) -> str:
            txt = re.sub(r'^\s*(?:["\'\:\-\s]+)?', "", txt.strip())
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt

        return _clean(part1), _clean(part2)

    def parse(self):
        content = self._read_and_normalize()
        reviews = content.split("### **Customer Review:**")[1:]
        parsed_rows = []

        for review_idx, review_block in enumerate(reviews, start=1):
            ts = self.base_time - timedelta(days=random.randint(0, self.max_days))
            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M:%S")
            customer_review = review_block.strip().split('\n', 1)[0].strip().strip('"')

            pattern = r"### \*\*1\. Experience Driver\*\*"
            matches = list(re.finditer(pattern, review_block))

            for entity_idx, match in enumerate(matches, start=1):
                start = match.start()
                end = matches[entity_idx].start() if entity_idx < len(matches) else len(review_block)
                entity_block = review_block[start:end]

                sas_full = self._extract_block("Semantic Action Statement", entity_block)
                sas_customer, sas_response = self._split_semantic_action(sas_full)

                row = {
                    "review_id": review_idx,
                    "entity_id": f"{review_idx}.{entity_idx}",
                    "date": date_str,
                    "time": time_str,
                    "customer_review": customer_review,

                    "experience_driver": self._extract_inline("Experience Driver", entity_block),
                    "entity_name": self._extract_inline("Entity Name", entity_block),
                    "theme": self._extract_inline("Theme", entity_block),
                    "context": self._extract_inline("Context", entity_block),
                    "feedback_type": self._extract_inline("Feedback Type", entity_block),
                    "emotion": self._extract_inline("Emotion", entity_block),
                    "customer_journey": self._extract_inline("Customer Journey", entity_block),
                    "customer_journey_stage": self._extract_inline("Customer Journey Stage", entity_block),
                    "interaction_moment": self._extract_inline("Interaction Moment", entity_block),
                    "opportunity_stream": self._extract_inline("Opportunity Maximisation Stream", entity_block),
                    "customer_effort_score": self._extract_inline("Customer Effort Score", entity_block, as_int=True),
                    "ou_name": self._extract_inline("OU Name", entity_block),
                    "keywords": self._extract_keywords(entity_block),

                    "semantic_action_statement": sas_full,
                    "semantic_customer_reality": sas_customer,
                    "semantic_strategic_response": sas_response,

                    "matters": self._extract_flexible("Matters", entity_block),
                    "stream_justification": self._extract_block("Stream Justification", entity_block),
                    "behavioral_impact": self._extract_block("Behavioral Impact", entity_block),
                }

                row["emotion_primary"], row["emotion_specific"] = self._split_emotion(row["emotion"])
                parsed_rows.append(row)

        self.output_df = pd.DataFrame(parsed_rows)

        # 🔧 CLEAN HERE
        self._post_cleanse()

    # ---------- Canonicalization helpers (static) ----------
    _FEEDBACK_MAP = {
        "complaint": "Complaint",
        "compliment": "Compliment",
        "request": "Request",
        "suggestion": "Suggestion",
        "question": "Question",
        "product usage insight": "Product Usage Insight",
        "usage insight": "Product Usage Insight",
        "emerging trends": "Emerging Trends / Market Insight",
        "market insight": "Emerging Trends / Market Insight",
        "emerging trends / market insight": "Emerging Trends / Market Insight",
    }

    _STREAM_MAP = {
        "fix": "Fix",
        "optimize": "Optimize",
        "optimise": "Optimize",
        "amplify": "Amplify",
        "innovate": "Innovate",
    }

    @staticmethod
    def _strip_zwc(s: str) -> str:
        return re.sub(r"[\u200B-\u200D\uFEFF]", "", str(s))

    @staticmethod
    def _canon_spaces(s: str) -> str:
        s = re.sub(r"\s+", " ", str(s))
        return s.strip()

    @staticmethod
    def _canon_arrow(s: str) -> str:
        s = str(s).replace("->", "→").replace("—>", "→").replace("-->", "→")
        s = re.sub(r"\s*→\s*", " → ", s)
        return s

    @staticmethod
    def _canon_text(s: str) -> str:
        s = unicodedata.normalize("NFKC", str(s))
        s = ReviewParser._strip_zwc(s)
        s = ReviewParser._canon_arrow(s)
        s = ReviewParser._canon_spaces(s)
        return s

    @staticmethod
    def _canon_feedback_type(s: str) -> str:
        k = ReviewParser._canon_text(s).lower()
        return ReviewParser._FEEDBACK_MAP.get(k, ReviewParser._canon_text(s))

    @staticmethod
    def _canon_stream(s: str) -> str:
        k = ReviewParser._canon_text(s).lower()
        return ReviewParser._STREAM_MAP.get(k, ReviewParser._canon_text(s))

    @staticmethod
    def _canon_experience_driver(s: str) -> str:
        return ReviewParser._canon_text(s)

    # ---------- Post-cleanse hook ----------
    def _post_cleanse(self):
        if self.output_df is None or self.output_df.empty:
            return

        df = self.output_df

        # 1) canonicalize key text fields
        text_cols = [
            "experience_driver","entity_name","theme","context","feedback_type",
            "emotion","customer_journey","customer_journey_stage","interaction_moment",
            "opportunity_stream","ou_name","matters","stream_justification","behavioral_impact",
            "semantic_action_statement","semantic_customer_reality","semantic_strategic_response",
            "customer_review"
        ]
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).map(self._canon_text)

        # 2) ED, Stream, Feedback Type canonical maps
        if "experience_driver" in df:
            df["experience_driver"] = df["experience_driver"].map(self._canon_experience_driver)
        if "opportunity_stream" in df:
            df["opportunity_stream"] = df["opportunity_stream"].map(self._canon_stream)
        if "feedback_type" in df:
            df["feedback_type"] = df["feedback_type"].map(self._canon_feedback_type)

        # 3) ensure date/time formats + optional timestamp
        if "date" in df and "time" in df:
            dt = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
            df["timestamp"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%H:%M:%S")

        # 4) keywords: ensure list-of-str, stripped
        if "keywords" in df.columns:
            def _clean_kw_list(x):
                if isinstance(x, list):
                    return [self._canon_text(k) for k in x]
                return x
            df["keywords"] = df["keywords"].map(_clean_kw_list)

        # 5) optional: drop exact duplicate OU rows
        dedupe_cols = [
            "experience_driver","entity_name","context","feedback_type",
            "customer_journey","customer_journey_stage","interaction_moment",
            "opportunity_stream","matters","semantic_action_statement"
        ]
        keep_cols = [c for c in dedupe_cols if c in df.columns]
        if keep_cols:
            df.drop_duplicates(subset=keep_cols, inplace=True, ignore_index=True)

        self.output_df = df

    def save(self, output_path: str = None) -> str:
        if self.output_df is None:
            raise ValueError("parse() must be run before save().")

        if not output_path:
            base_name = self.csv_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_flattened_v2.csv"

        self.output_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Flattened output saved to: {output_path}")
        return output_path

