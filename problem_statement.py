import sqlite3
import pandas as pd
from typing import Dict
from openai import OpenAI
import os
import re

# 🔐 API Client Setup
def get_client():
    api_key = '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'
    return OpenAI(api_key=api_key)

# 📤 Prompt Caller
def get_response(system_prompt: str, user_prompt: str) -> Dict:
    client = get_client()
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )
    return response.model_dump()

# 📦 Cluster Loader
def load_cluster_from_db(group_id: str, db_path: str = "outputs/clusters.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM clusters WHERE bcs_group_id = ?"
    df = pd.read_sql_query(query, conn, params=(group_id,))
    conn.close()
    return df


# 🚀 Main Synthesis Function
def synthesize_and_update_cluster(group_id: str, system_prompt: str, db_path: str = "outputs/clusters.db"):
    df = load_cluster_from_db(group_id, db_path)
    if df.empty:
        print(f"⚠️ No records found for group_id: {group_id}")
        return

    # Convert full cluster to CSV string
    csv_string = df.to_csv(index=False)

    # 🔁 Send to LLM
    response = get_response(system_prompt, csv_string)

    # Extract LLM message content only
    choices = response.get('choices', [])
    if not choices:
        print("⚠️ No response from LLM.")
        return

    message = choices[0].get('message', {})
    content = message.get('content', '')
    if not content:
        print("⚠️ Empty message content from LLM.")
        return

    # 💾 Add raw content into a column for manual or later parsing
    df["raw_response"] = content

    # 🔬 Save debug output
    debug_path = f"data/cluster_debug_{group_id}.csv"
    df.to_csv(debug_path, index=False)
    print(f"🧪 Raw LLM output saved → {debug_path}")

    print(f"🧠 Cluster {group_id} synthesis complete.")
    return df

# 🧾 Prompt Loader
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 🔁 Main
def main():
    group_id = "delivery_agi_fix_c809af78"
    prompt_path = "prompt/pdca_problem_statement_prompt.txt"
    system_prompt = load_prompt_from_file(prompt_path)

    synthesize_and_update_cluster(
        group_id=group_id,
        system_prompt=system_prompt,
        db_path="outputs/clusters.db"
    )

if __name__ == "__main__":
    main()



# import sqlite3
# import pandas as pd
# from typing import Dict

# import sqlite3
# import pandas as pd
# from typing import Dict
# from openai import OpenAI
# import os

# # ─────────── 🔐 API Client Setup ───────────
# def get_client():
#     api_key = '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'
#     client = OpenAI(api_key=api_key)
#     return client

# # ─────────── 🎤 Prompt Caller ───────────
# def get_response(system_prompt: str, user_prompt: str) -> Dict:
#     client = get_client()
#     messages = [
#         {'role': 'system', 'content': system_prompt},
#         {'role': 'user', 'content': user_prompt},
#     ]
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         temperature=0.0
#     )
#     return response.model_dump()

# # ─────────── 📦 Cluster DB I/O ───────────
# def load_cluster_from_db(group_id: str, db_path: str = "outputs/clusters.db") -> pd.DataFrame:
#     conn = sqlite3.connect(db_path)
#     query = "SELECT * FROM clusters WHERE bcs_group_id = ?"
#     df = pd.read_sql_query(query, conn, params=(group_id,))
#     conn.close()
#     return df

# # def update_cluster_rows_in_db(df: pd.DataFrame, db_path: str = "outputs/clusters.db"):
# #     conn = sqlite3.connect(db_path)
# #     cursor = conn.cursor()

# #     for _, row in df.iterrows():
# #         cursor.execute('''
# #             UPDATE clusters
# #             SET
# #                 semantic_action_statement = ?,
# #                 problem_statement = ?,
# #                 matters = ?,
# #                 context = ?,
# #                 interaction_moment = ?
# #             WHERE bcs_id = ?
# #         ''', (
# #             row["semantic_action_statement"],
# #             row["problem_statement"],
# #             row["matters"],
# #             row["context"],
# #             row["interaction_moment"],
# #             row["bcs_id"]
# #         ))

# #     conn.commit()
# #     conn.close()
# #     print(f"✅ Updated {len(df)} rows with synthesized fields.")

# # ─────────── 🧠 Core Synthesis Orchestrator ───────────
# def synthesize_and_update_cluster(group_id: str, system_prompt: str, db_path: str = "outputs/clusters.db"):
#     df = load_cluster_from_db(group_id, db_path)
#     if df.empty:
#         print(f"⚠️ No records found for group_id: {group_id}")
#         return

#     # Convert full cluster to CSV string
#     csv_string = df.to_csv(index=False)

#     # 🔁 Send to LLM
#     response = get_response(system_prompt, csv_string)

#     # Extract LLM content
#     choices = response.get('choices', [])
#     if not choices:
#         print("⚠️ No response from LLM.")
#         return

#     content = choices[0].get('message', {}).get('content', '')
#     if not content:
#         print("⚠️ Empty message content from LLM.")
#         return

    
#         # 🧠 Parse structured YAML-style string from LLM
#     import yaml
#     try:
#         parsed = yaml.safe_load(content)
#     except yaml.YAMLError as e:
#         print("❌ YAML parsing failed.")
#         print(content)
#         raise e

#     # 💾 Save parsed content for inspection
#     for field in ["semantic_action_statement", "problem_statement", "matters", "context", "interaction_moment"]:
#         df[field] = parsed.get(field, "")

#     # 🔬 Save debug output
#     debug_path = f"data/cluster_debug_{group_id}.csv"
#     df.to_csv(debug_path, index=False)
#     print(f"🧪 Debug CSV saved → {debug_path}")

#     # (DB update currently disabled)
#     print(f"🧠 Cluster {group_id} synthesis complete.")
#     return df


#     # # 🧩 Inject the five extracted fields into all rows
#     # for field in ["semantic_action_statement", "problem_statement", "matters", "context", "interaction_moment"]:
#     #     df[field] = parsed.get(field, "")

#     # # 💾 Push into DB
#     # update_cluster_rows_in_db(df, db_path)

#     print(f"🧠 Cluster {group_id} synthesis complete.")
#     return df

# # Load system prompt from file
# def load_prompt_from_file(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()

# # Run the synthesis pipeline
# group_id = "delivery_agi_fix_c809af78"
# prompt_path = "prompt/pdca_problem_statement_prompt.txt"
# system_prompt = load_prompt_from_file(prompt_path)

# # Run it
# synthesize_and_update_cluster(group_id=group_id, system_prompt=system_prompt, db_path="outputs/clusters.db")


