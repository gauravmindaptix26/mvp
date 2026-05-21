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

@app.route("/debug")
def debug():
    import sys
    key = os.getenv("OPENAI_API_KEY")
    excel_path = os.path.join(BASE_DIR, "creators.xlsx")
    return jsonify({
        "api_key_set": bool(key),
        "api_key_prefix": key[:8] if key else None,
        "excel_exists": os.path.exists(excel_path),
        "excel_path": excel_path,
        "df_rows": len(df) if df is not None else 0,
        "python": sys.version
    })

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query", "")

        filter_prompt = f"""
Extract search filters from this influencer search query.
Query: {query}

Excel columns available:
- Name (influencer name)
- Followers (number)
- Category (e.g. Trading, Business and Marketing, Personal finance, Career)
- Location (city name)
- Reel + story reshare (price in rupees)
- Story (price in rupees)
- Static Post (price in rupees)
- UGC (Social media but No Ads) (price in rupees)
- UGC with 1 Month RIghts (price in rupees)
- 1 Month Digital RIghts (price in rupees)

Return JSON only with these keys (use null if not mentioned):
{{
  "name": "",
  "category": "",
  "location": "",
  "min_followers": null,
  "max_followers": null,
  "max_reel_price": null,
  "max_story_price": null,
  "max_static_price": null,
  "max_ugc_price": null,
  "max_digital_rights_price": null
}}
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

        if filters.get("name"):
            filtered = filtered[filtered["Name"].astype(str).str.contains(filters["name"], case=False, na=False)]

        if filters.get("location"):
            filtered = filtered[filtered["Location"].astype(str).str.contains(filters["location"], case=False, na=False)]

        if filters.get("category"):
            filtered = filtered[filtered["Category"].astype(str).str.contains(filters["category"], case=False, na=False)]

        if filters.get("min_followers"):
            filtered = filtered[pd.to_numeric(filtered["Followers"], errors="coerce") >= filters["min_followers"]]

        if filters.get("max_followers"):
            filtered = filtered[pd.to_numeric(filtered["Followers"], errors="coerce") <= filters["max_followers"]]

        if filters.get("max_reel_price"):
            filtered = filtered[pd.to_numeric(filtered["Reel + story reshare"], errors="coerce") <= filters["max_reel_price"]]

        if filters.get("max_story_price"):
            filtered = filtered[pd.to_numeric(filtered["Story"], errors="coerce") <= filters["max_story_price"]]

        if filters.get("max_static_price"):
            filtered = filtered[pd.to_numeric(filtered["Static Post"], errors="coerce") <= filters["max_static_price"]]

        if filters.get("max_ugc_price"):
            filtered = filtered[pd.to_numeric(filtered["UGC (Social media but No Ads)"], errors="coerce") <= filters["max_ugc_price"]]

        if filters.get("max_digital_rights_price"):
            filtered = filtered[pd.to_numeric(filtered["1 Month Digital RIghts"], errors="coerce") <= filters["max_digital_rights_price"]]

        filtered = filtered.head(111)
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
