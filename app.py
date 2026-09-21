"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that estimates annual cost for a single
policyholder using the fields an underwriting system would have at
application time. The cleaning and feature-engineering logic from
clean_and_engineer.py is reused here rather than duplicated, so a
transformation change only ever needs to happen in one place.

Run with:  uvicorn app:app --reload
"""

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Policyholder Cost Estimator API", version="1.0")

MODEL = joblib.load("outputs/cost_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")


class ApplicantRecord(BaseModel):
    age: int
    sex: str  # "male" | "female"
    bmi: float
    children: int
    smoker: str  # "yes" | "no"
    region: str  # "northeast" | "northwest" | "southeast" | "southwest"


class CostEstimateResponse(BaseModel):
    estimated_annual_cost: float
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Policyholder Cost Estimator</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>Policyholder Cost Estimator — Test Form</h2>
        <p>Every field below maps 1:1 to a field in <code>ApplicantRecord</code>
           (app.py) — nothing here is hardcoded in the page itself. Full API
           docs at <a href="/docs">/docs</a>.</p>
        <form id="costForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Age<br><input name="age" type="number" value="45" required></label>
            <label>BMI<br><input name="bmi" type="number" step="0.1" value="32" required></label>
            <label>Children<br><input name="children" type="number" value="2" required></label>
            <label>Sex<br>
                <select name="sex" required><option>female</option><option>male</option></select>
            </label>
            <label>Smoker?<br>
                <select name="smoker" required><option selected>yes</option><option>no</option></select>
            </label>
            <label>Region<br>
                <select name="region" required>
                    <option>northeast</option><option>northwest</option>
                    <option>southeast</option><option>southwest</option>
                </select>
            </label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Estimate cost</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("costForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                age: parseInt(form.get("age"), 10),
                sex: form.get("sex"),
                bmi: parseFloat(form.get("bmi")),
                children: parseInt(form.get("children"), 10),
                smoker: form.get("smoker"),
                region: form.get("region")
            };
            const res = await fetch("/estimate", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                document.getElementById("result").innerText =
                    "Error " + res.status + ": " + await res.text();
                return;
            }
            const data = await res.json();
            document.getElementById("result").innerText =
                "Estimated annual cost: $" + data.estimated_annual_cost.toLocaleString() +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """


@app.post("/estimate", response_model=CostEstimateResponse)
def estimate_cost(record: ApplicantRecord):
    raw_row = record.model_dump()
    raw_row["charges"] = 0  # placeholder; unused for scoring, but clean_data()/engineer_features() expect the column
    raw_df = pd.DataFrame([raw_row])

    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)

    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred_log_cost = MODEL.predict(X)[0]
    pred_cost = float(np.expm1(pred_log_cost))

    importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return CostEstimateResponse(
        estimated_annual_cost=round(pred_cost, 2),
        top_factors=top_factors,
    )
