# =========================================================
# CREDITSENTINEL FASTAPI - RENDER DEPLOYMENT VERSION
# =========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import traceback
import os
import math

from typing import List, Optional

from feature_engine import compute_features

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="CreditSentinel API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# BASE DIRECTORY & LOAD FILES
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "lightgbm_0.8106.pkl"))
print("✅ Model Loaded")

applications_df = pd.read_csv(os.path.join(BASE_DIR, "loan_applications.csv"))
print(f"✅ Applications Loaded: {len(applications_df)} rows")

# Print all columns at startup so you can see them in Render logs
print("CSV COLUMNS:", list(applications_df.columns))

# =========================================================
# AUTO-DETECT CIBIL COLUMN
# Works in 3 passes — handles ANY column name in your CSV.
# =========================================================
CIBIL_COLUMN = None

# Pass 1 — exact name match
for candidate in [
    "cibil_score", "credit_score", "cibil", "bureau_score", "creditScore",
    "CIBIL Score", "CIBIL_score", "Credit Score", "bureau_credit_score",
    "fico_score", "credit_rating", "CreditScore", "CREDIT_SCORE", "CIBIL",
    "BureauScore", "cibil score", "Credit_Score", "bureau_cibil_score",
    "cibil_rating", "credit_bureau_score", "cb_score", "external_score",
    "bureau_rating", "risk_score_bureau"
]:
    if candidate in applications_df.columns:
        CIBIL_COLUMN = candidate
        print(f"✅ CIBIL column (exact match): '{CIBIL_COLUMN}'")
        break

# Pass 2 — case-insensitive keyword scan
if CIBIL_COLUMN is None:
    for col in applications_df.columns:
        col_norm = col.lower().replace(" ", "_").replace("-", "_")
        if any(kw in col_norm for kw in ["cibil", "bureau", "fico", "creditscore", "credit_score", "credit_rating"]):
            CIBIL_COLUMN = col
            print(f"✅ CIBIL column (keyword scan): '{CIBIL_COLUMN}'")
            break

# Pass 3 — value-range heuristic
# Finds any numeric column where ≥70% of values are between 300–900.
# This catches your column regardless of what it is named.
if CIBIL_COLUMN is None:
    print("Passes 1 & 2 failed — running value-range scan across all columns...")
    exclude = ["income", "amount", "emi", "loan", "salary", "age", "tenure", "month", "year", "id"]
    best_col, best_pct = None, 0.0
    for col in applications_df.columns:
        if any(kw in col.lower() for kw in exclude):
            continue
        series = pd.to_numeric(applications_df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        pct = ((series >= 300) & (series <= 900)).sum() / len(series)
        median = series.median()
        print(f"   '{col}' → pct_in_300_900={pct:.2f}, median={median:.0f}")
        if pct >= 0.70 and 300 <= median <= 900 and pct > best_pct:
            best_col, best_pct = col, pct
    if best_col:
        CIBIL_COLUMN = best_col
        print(f"✅ CIBIL column (value-range heuristic): '{CIBIL_COLUMN}'")

if CIBIL_COLUMN is None:
    print("❌ CIBIL column NOT found. Open Render logs and look for the column")
    print("   printed above whose values are between 300–900, then hard-code it")
    print("   by adding its name at the top of the exact-match list in Pass 1.")
else:
    print(f"   Sample CIBIL values: {applications_df[CIBIL_COLUMN].dropna().head(5).tolist()}")

# =========================================================
# MODEL FEATURES
# =========================================================
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = list(model.feature_name_())

# =========================================================
# SAFE HELPERS
# =========================================================
def safe_float(val, default=0.0):
    try:
        result = float(val)
        return default if (math.isnan(result) or math.isinf(result)) else result
    except:
        return default

def safe_int(val, default=0):
    try:
        result = float(val)
        return default if (math.isnan(result) or math.isinf(result)) else int(result)
    except:
        return default

def safe_str(val, default=""):
    try:
        if val is None: return default
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return default
        return str(val)
    except:
        return default

# =========================================================
# SHARED HELPERS
# =========================================================
def get_risk_tier(risk_score: float) -> str:
    if risk_score < 0.4:
        return "Low"
    elif risk_score < 0.65:
        return "Medium"
    else:
        return "High"

def get_credit_score_from_risk(risk_score: float) -> int:
    return int(300 + (1 - risk_score) * 600)

def get_status(risk_tier: str) -> str:
    return {
        "Low":    "Approved",
        "Medium": "Under Review",
        "High":   "Rejected"
    }.get(risk_tier, "Pending")

def get_foir(monthly_income: float, monthly_emi: float) -> float:
    return round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0.0

# =========================================================
# CIBIL SCORE — reads directly from CSV row
# Returns 0 only if the column genuinely has no value.
# Never falls back to ML-derived value (kept separate).
# =========================================================
def get_raw_cibil_score(row) -> int:
    if CIBIL_COLUMN is None:
        return 0
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    val = row_dict.get(CIBIL_COLUMN, None)
    if val is None:
        return 0
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0
    try:
        return max(int(float(val)), 0)
    except (ValueError, TypeError):
        return 0

# ML-derived credit score (300–900), always has a value
def get_ml_credit_score(risk_score: float) -> int:
    return get_credit_score_from_risk(risk_score)

# =========================================================
# CORE: RUN ML MODEL
# =========================================================
def generate_risk_score(application_id: str) -> dict:
    try:
        features_dict     = compute_features(application_id)
        filtered_features = {f: features_dict.get(f, 0) for f in MODEL_FEATURES}
        features_df       = pd.DataFrame([filtered_features])[MODEL_FEATURES]
        features_df       = features_df.fillna(0).replace([np.inf, -np.inf], 0).astype(float)
        risk_score = round(float(model.predict_proba(features_df)[:, 1][0]), 4)
        risk_tier  = get_risk_tier(risk_score)
        return {"risk_score": risk_score, "risk_tier": risk_tier}
    except Exception as e:
        print(traceback.format_exc())
        return {"risk_score": 0.0, "risk_tier": "Low"}

# =========================================================
# REQUEST MODELS
# =========================================================
class ScoreRequest(BaseModel):
    application_id: str

class BatchScoreRequest(BaseModel):
    application_ids: List[str]

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status":               "ok",
        "model_loaded":         True,
        "total_applications":   len(applications_df),
        "cibil_column":         CIBIL_COLUMN or "NOT_FOUND"
    }

# =========================================================
# SCORE SINGLE
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):
    try:
        result = generate_risk_score(req.application_id)
        return {
            "application_id": req.application_id,
            "model_loaded":   True,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"],
            "features_used":  len(MODEL_FEATURES)
        }
    except Exception as e:
        return {"application_id": req.application_id, "model_loaded": False, "error": str(e)}

# =========================================================
# SCORE BATCH
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    results = []
    for app_id in req.application_ids:
        result = generate_risk_score(app_id)
        results.append({
            "application_id": app_id,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"]
        })
    return {"total_applications": len(results), "results": results}

# =========================================================
# APPLICATIONS LIST
# =========================================================
@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    try:
        applications = []
        subset = applications_df.iloc[offset: offset + limit]

        for _, row in subset.iterrows():
            app_id         = safe_str(row.get("application_id", ""))
            result         = generate_risk_score(app_id)
            risk_score     = result["risk_score"]
            risk_tier      = result["risk_tier"]
            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

            applications.append({
                "application_id":     app_id,
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         risk_score,
                "risk_tier":          risk_tier,
                "cibil_score":        get_raw_cibil_score(row),        # real CSV value
                "credit_score":       get_ml_credit_score(risk_score), # ML-derived 300-900
                "application_status": get_status(risk_tier)
            })

        return {"total": len(applications_df), "applications": applications}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# APPLICATION DETAIL
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):
    try:
        matched = applications_df[
            applications_df["application_id"].astype(str) == str(application_id)
        ]
        if len(matched) == 0:
            return {"error": "Application not found"}

        row            = matched.iloc[0]
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))
        foir           = round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0

        score_data   = generate_risk_score(application_id)
        risk_score   = score_data["risk_score"]
        risk_tier    = score_data["risk_tier"]
        cibil_score  = get_raw_cibil_score(row)        # real CSV bureau score (573, 706 etc.)
        credit_score = get_ml_credit_score(risk_score) # ML-derived 300-900

        application_status = safe_str(row.get("application_status", row.get("status", "")))
        if application_status == "":
            if risk_score >= 0.75:   application_status = "Rejected"
            elif risk_score >= 0.45: application_status = "Pending"
            else:                    application_status = "Approved"

        print(f"[DETAIL] id={application_id} | col='{CIBIL_COLUMN}' | cibil={cibil_score} | credit={credit_score} | risk={risk_score}")

        return {
            "application_id":     safe_str(row.get("application_id", "")),
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", row.get("loan_amount", 0))),
            "foir":               foir,
            "cibil_score":        cibil_score,
            "credit_score":       credit_score,
            "risk_score":         risk_score,
            "risk_tier":          risk_tier,
            "application_status": application_status,
            "date_applied":       safe_str(row.get("date_applied", row.get("created_at", "")))
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        high = medium = low = 0
        sample_df = applications_df.sample(n=min(500, len(applications_df)), random_state=42)

        for _, row in sample_df.iterrows():
            app_id = safe_str(row.get("application_id", ""))
            tier   = generate_risk_score(app_id)["risk_tier"]
            if tier == "High":     high   += 1
            elif tier == "Medium": medium += 1
            else:                  low    += 1

        total = len(applications_df)
        scale = total / len(sample_df)

        return {
            "total_applications": total,
            "high":   round(high   * scale),
            "medium": round(medium * scale),
            "low":    round(low    * scale)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

