import re
import json
import random
from datetime import datetime, timedelta
from typing import List, Optional, Union
import pandas as pd


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
            if val.lower() == "null":
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

    # <<< FIX: NEW ROBUST FUNCTION FOR INCONSISTENT FIELDS
    def _extract_flexible(self, label: str, block: str) -> str:
        """
        Extracts a field value that may or may not be quoted and is case-insensitive.
        Handles both `**Label**: "Value"` and `**label**: Value`.
        """
        # Pattern explained:
        # - \*\*Label\*\*:  -> Matches the label, case-insensitively
        # \s*             -> Optional whitespace
        # (")?            -> Optionally captures a quote (Group 1)
        # (.+?)           -> The actual content (Group 2)
        # \1?             -> Optionally matches the first captured group (the quote).
        #                   This ensures if there's an opening quote, there must be a closing one.
        # (?=...)         -> Positive lookahead for the end of the field.
        pattern = rf'- \*\*{label}\*\*:\s*(")?(.+?)\1?(?=\n- \*\*|---|###|\Z)'
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        if match:
            # The actual content is always in group 2
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

                    "semantic_action": self._extract_block("Semantic Action Statement", entity_block),
                    # <<< FIX: Using the new flexible function for the 'Matters' field
                    "matters": self._extract_flexible("Matters", entity_block),
                    "stream_justification": self._extract_block("Stream Justification", entity_block),
                    "behavioral_impact": self._extract_block("Behavioral Impact", entity_block),
                }

                row["emotion_primary"], row["emotion_specific"] = self._split_emotion(row["emotion"])
                parsed_rows.append(row)

        self.output_df = pd.DataFrame(parsed_rows)

    def save(self, output_path: str = None) -> str:
        if self.output_df is None:
            raise ValueError("parse() must be run before save().")

        if not output_path:
            base_name = self.csv_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_flattened.csv"

        self.output_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Flattened output saved to: {output_path}")
        return output_path
