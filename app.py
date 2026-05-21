import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

df = pd.read_excel("creators.xlsx", header=1)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():

    query = request.json["query"]

    # Step 1: Extract filters using OpenAI
    filter_prompt = f"""
Extract influencer search filters from this query.

Query:
{query}

Return JSON only.

Example:
{{
  "city":"",
  "category":"",
  "max_budget":null,
  "min_followers":null
}}
"""

    filter_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a JSON extractor. Return only valid JSON, no markdown, no explanation."},
            {"role":"user","content":filter_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = filter_response.choices[0].message.content.strip()
    filters = json.loads(raw)

    filtered = df.copy()

    # City
    city = filters.get("city") or filters.get("location") or ""
    if city:
        filtered = filtered[
            filtered["Location"]
            .astype(str)
            .str.contains(city, case=False, na=False)
        ]

    # Category
    if filters.get("category"):
        filtered = filtered[
            filtered["Category"]
            .astype(str)
            .str.contains(
                filters["category"],
                case=False,
                na=False
            )
        ]

    # Followers
    if filters.get("min_followers"):
        filtered = filtered[
            pd.to_numeric(
                filtered["Followers"],
                errors="coerce"
            ) >= filters["min_followers"]
        ]

    # Budget
    if filters.get("max_budget"):
        filtered = filtered[
            pd.to_numeric(
                filtered["Reel + story reshare"],
                errors="coerce"
            ) <= filters["max_budget"]
        ]

    filtered = filtered.head(10)

    print("=== FILTERS:", filters)
    print("=== RESULTS COUNT:", len(filtered))
    print("=== COLUMNS:", df.columns.tolist())

    creators = filtered.fillna("").to_dict(orient="records")

    # Step 2: Natural GPT Response
    summary_prompt = f"""
User asked:

{query}

Matching creators:

{creators}

Write a professional response like ChatGPT.

Requirements:
- Mention total creators found.
- Explain why they match.
- Recommend best options.
- Keep under 150 words.
"""

    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"user",
                "content":summary_prompt
            }
        ]
    )

    summary = (
        summary_response
        .choices[0]
        .message
        .content
    )

    return jsonify({
        "summary": summary,
        "results": creators
    })

if __name__ == "__main__":
    app.run(debug=True)