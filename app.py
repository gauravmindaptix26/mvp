import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(BASE_DIR, "creators.xlsx"), header=1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query", "")

        filter_prompt = f"""
Extract influencer search filters from this query.
Query: {query}
Return JSON only with these keys: city, category, max_budget, min_followers
Example: {{"city":"","category":"","max_budget":null,"min_followers":null}}
"""
        filter_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON extractor. Return only valid JSON, no markdown."},
                {"role": "user", "content": filter_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        filters = json.loads(filter_response.choices[0].message.content.strip())
        filtered = df.copy()

        city = filters.get("city") or filters.get("location") or ""
        if city:
            filtered = filtered[
                filtered["Location"].astype(str).str.contains(city, case=False, na=False)
            ]

        category = filters.get("category") or ""
        if category:
            filtered = filtered[
                filtered["Category"].astype(str).str.contains(category, case=False, na=False)
            ]

        min_followers = filters.get("min_followers")
        if min_followers:
            filtered = filtered[
                pd.to_numeric(filtered["Followers"], errors="coerce") >= min_followers
            ]

        max_budget = filters.get("max_budget")
        if max_budget:
            filtered = filtered[
                pd.to_numeric(filtered["Reel + story reshare"], errors="coerce") <= max_budget
            ]

        filtered = filtered.head(10)
        creators = filtered.fillna("").to_dict(orient="records")

        summary_prompt = f"""
User asked: {query}
Matching creators: {creators}
Write a professional 2-3 line response. Mention total found and best recommendations.
"""
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = summary_response.choices[0].message.content

        return jsonify({"summary": summary, "results": creators})

    except Exception as e:
        return jsonify({"error": str(e), "summary": f"Error: {str(e)}", "results": []}), 500

if __name__ == "__main__":
    app.run(debug=True)
