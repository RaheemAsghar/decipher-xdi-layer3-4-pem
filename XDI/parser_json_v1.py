import json
import re
import unicodedata
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime


class ReviewParserJSON:
    """Parser for JSON-formatted CX experience driver outputs."""

    def __init__(self, input_path: str):
        self.csv_path: str = input_path
        self.output_df: Optional[pd.DataFrame] = None

    def _parse_json_safe(self, json_str: str) -> Optional[Dict[Any, Any]]:
        """Safely parse JSON string, handling common issues."""
        try:
            # Clean up potential escaping issues
            json_str = str(json_str).strip()
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            return None

    def _extract_processed_at(self, timestamp_str: str) -> tuple[str, str]:
        """Extract date and time from ISO timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except Exception:
            return "", ""

    def parse(self):
        """Parse CSV containing JSON outputs and flatten to relational structure."""
        # Read CSV without header - first column contains JSON
        df_input = pd.read_csv(self.csv_path, header=None, encoding="utf-8")
        
        parsed_rows = []
        review_counter = 1

        for idx, row in df_input.iterrows():
            json_str = row.iloc[0]  # First column
            
            # Parse JSON
            data = self._parse_json_safe(json_str)
            if data is None:
                continue
            
            # FAILURE CASE: Skip entirely
            if data.get("status") == "skipped":
                print(f"⏭️  Skipping review {review_counter} (insufficient_info)")
                continue
            
            # SUCCESS CASE: Process experience drivers
            schema_version = data.get("schema_version", "")
            processed_at = data.get("processed_at", "")
            customer_review = data.get("customer_review", "")
            
            date_str, time_str = self._extract_processed_at(processed_at)
            
            experience_drivers = data.get("experience_drivers", [])
            
            if not experience_drivers:
                print(f"⚠️  Review {review_counter} has no experience drivers")
                continue
            
            # Create one row per experience driver
            for entity_idx, driver in enumerate(experience_drivers, start=1):
                # Extract nested structures
                ed = driver.get("experience_driver", {})
                context = driver.get("context", {})
                emotion = driver.get("emotion", {})
                journey = driver.get("journey", {})
                ou = driver.get("orchestration_unit", {})
                sas = ou.get("semantic_action_statement", {})
                
                row_data = {
                    # Common fields
                    "review_id": review_counter,
                    "entity_id": f"{review_counter}.{entity_idx}",
                    "schema_version": schema_version,
                    "date": date_str,
                    "time": time_str,
                    "customer_review": customer_review,
                    
                    # Experience Driver taxonomy
                    "experience_driver_label": ed.get("label", ""),
                    "experience_driver_category": ed.get("category", ""),
                    "experience_driver_subcategory": ed.get("subcategory", ""),
                    "experience_driver": f"{ed.get('category', '')} → {ed.get('subcategory', '')}",
                    
                    # Core driver fields
                    "theme": driver.get("theme", ""),
                    "entity_name": driver.get("entity_name", ""),
                    "context": context.get("text", ""),
                    "keywords": context.get("keywords", []),
                    "feedback_type": driver.get("feedback_type", ""),
                    
                    # Emotion
                    "emotion_primary": emotion.get("group", ""),
                    "emotion_specific": emotion.get("specific", ""),
                    "emotion": f"{emotion.get('group', '')} - {emotion.get('specific', '')}",
                    
                    # Journey
                    "customer_journey": journey.get("customer_journey", ""),
                    "customer_journey_stage": journey.get("customer_journey_stage", ""),
                    "interaction_moment": journey.get("interaction_moment", ""),
                    
                    # Stream & Effort
                    "opportunity_stream": driver.get("opportunity_maximisation_stream", ""),
                    "customer_effort_score": driver.get("customer_effort_score", None),
                    
                    # Orchestration Unit
                    "ou_name": ou.get("ou_name", ""),
                    "matters": ou.get("matters", ""),
                    "stream_justification": ou.get("stream_justification", ""),
                    "matters_extraction": ou.get("matters_extraction", ""),
                    "behavioral_impact": ou.get("behavioral_impact", ""),
                    
                    # Semantic Action Statement
                    "semantic_customer_reality": sas.get("section_1_customer_reality", ""),
                    "semantic_strategic_response": sas.get("section_2_strategic_response", ""),
                    "semantic_action_statement": f"{sas.get('section_1_customer_reality', '')} | {sas.get('section_2_strategic_response', '')}",
                }
                
                parsed_rows.append(row_data)
            
            review_counter += 1
        
        self.output_df = pd.DataFrame(parsed_rows)
        
        # Post-cleanse
        self._post_cleanse()
        
        print(f"✅ Parsed {len(parsed_rows)} experience drivers from {review_counter - 1} reviews")

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
        s = str(s).replace("->", "→").replace("â€">", "→").replace("-->", "→")
        s = re.sub(r"\s*→\s*", " → ", s)
        return s

    @staticmethod
    def _canon_text(s: str) -> str:
        s = unicodedata.normalize("NFKC", str(s))
        s = ReviewParserJSON._strip_zwc(s)
        s = ReviewParserJSON._canon_arrow(s)
        s = ReviewParserJSON._canon_spaces(s)
        return s

    @staticmethod
    def _canon_feedback_type(s: str) -> str:
        k = ReviewParserJSON._canon_text(s).lower()
        return ReviewParserJSON._FEEDBACK_MAP.get(k, ReviewParserJSON._canon_text(s))

    @staticmethod
    def _canon_stream(s: str) -> str:
        k = ReviewParserJSON._canon_text(s).lower()
        return ReviewParserJSON._STREAM_MAP.get(k, ReviewParserJSON._canon_text(s))

    @staticmethod
    def _canon_experience_driver(s: str) -> str:
        return ReviewParserJSON._canon_text(s)

    def _post_cleanse(self):
        """Apply canonicalization and cleanup to parsed data."""
        if self.output_df is None or self.output_df.empty:
            return

        df = self.output_df

        # 1) Canonicalize text fields
        text_cols = [
            "experience_driver", "experience_driver_label", "experience_driver_category",
            "experience_driver_subcategory", "entity_name", "theme", "context", "feedback_type",
            "emotion", "emotion_primary", "emotion_specific", "customer_journey",
            "customer_journey_stage", "interaction_moment", "opportunity_stream",
            "ou_name", "matters", "stream_justification", "behavioral_impact",
            "semantic_action_statement", "semantic_customer_reality",
            "semantic_strategic_response", "customer_review", "matters_extraction"
        ]
        
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).map(self._canon_text)

        # 2) Apply canonical mappings
        if "experience_driver" in df:
            df["experience_driver"] = df["experience_driver"].map(self._canon_experience_driver)
        if "opportunity_stream" in df:
            df["opportunity_stream"] = df["opportunity_stream"].map(self._canon_stream)
        if "feedback_type" in df:
            df["feedback_type"] = df["feedback_type"].map(self._canon_feedback_type)

        # 3) Ensure date/time formats + optional timestamp
        if "date" in df and "time" in df:
            dt = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
            df["timestamp"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")

        # 4) Keywords: ensure list-of-str, stripped
        if "keywords" in df.columns:
            def _clean_kw_list(x):
                if isinstance(x, list):
                    return [self._canon_text(k) for k in x]
                return x
            df["keywords"] = df["keywords"].map(_clean_kw_list)

        # 5) Deduplicate exact OU rows
        dedupe_cols = [
            "experience_driver", "entity_name", "context", "feedback_type",
            "customer_journey", "customer_journey_stage", "interaction_moment",
            "opportunity_stream", "matters", "semantic_action_statement"
        ]
        keep_cols = [c for c in dedupe_cols if c in df.columns]
        if keep_cols:
            df.drop_duplicates(subset=keep_cols, inplace=True, ignore_index=True)

        self.output_df = df

    def save(self, output_path: str = None) -> str:
        """Save parsed dataframe to CSV."""
        if self.output_df is None:
            raise ValueError("parse() must be run before save().")

        if not output_path:
            base_name = self.csv_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_flattened_v3.csv"

        self.output_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Flattened output saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser_json_v3.py <input_csv_path> [output_csv_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    parser = ReviewParserJSON(input_path)
    parser.parse()
    parser.save(output_path)
