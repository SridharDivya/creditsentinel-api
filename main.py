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

from typing import List

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

model = joblib.load(os.path.join(BASE_DIR, "creditsentinel_model_v1.pkl"))
print("✅ Model Loaded")

applications_df = pd.read_csv(os.path.join(BASE_DIR, "loan_applications.csv"))
print(f"✅ Applications Loaded: {len(applications_df)} rows")

# =========================================================
# MODEL FEATURES (resolved once at startup)
# =========================================================
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = list(model.feature_name_)

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

def get_credit_score(risk_score: float) -> int:
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
# CORE: RUN ML MODEL FOR ONE APPLICATION
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
        "status": "ok",
        "model_loaded": True,
        "total_applications": len(applications_df)
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
# APPLICATIONS LIST — BATCH ML (FAST)
# =========================================================
@app.get("/api/applications")
def get_applications():
    try:
        app_ids = applications_df["application_id"].tolist()

        # ── Batch compute all features at once ────────────
        all_features = compute_features_batch(app_ids)

        rows = []
        for app_id in app_ids:
            f = all_features.get(app_id, {})
            rows.append({feat: f.get(feat, 0) for feat in MODEL_FEATURES})

        # ── Single model call for ALL apps ────────────────
        features_df = pd.DataFrame(rows)[MODEL_FEATURES]
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

        risk_scores = model.predict_proba(features_df)[:, 1]

        applications = []
        for i, (_, row) in enumerate(applications_df.iterrows()):
            risk_score = round(float(risk_scores[i]), 4)
            risk_tier  = get_risk_tier(risk_score)

            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

            applications.append({
                "application_id":     safe_str(row.get("application_id", "")),
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         risk_score,
                "risk_tier":          risk_tier,
                "credit_score":       get_credit_score(risk_score),
                "application_status": get_status(risk_tier)
            })

        return {"total": len(applications), "applications": applications}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# APPLICATION DETAIL
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):
    try:
        if "application_id" not in applications_df.columns:
            return {"error": "application_id column missing"}

        matched = applications_df[applications_df["application_id"] == application_id]
        if len(matched) == 0:
            return {"error": "Application not found"}

        row            = matched.iloc[0]
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

        result     = generate_risk_score(application_id)
        risk_score = result["risk_score"]
        risk_tier  = result["risk_tier"]

        return {
            "application_id":     safe_str(row.get("application_id", "")),
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
            "foir":               get_foir(monthly_income, monthly_emi),
            "risk_score":         risk_score,
            "risk_tier":          risk_tier,
            "credit_score":       get_credit_score(risk_score),
            "application_status": get_status(risk_tier)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# PORTFOLIO SUMMARY — FIXED (no db, uses CSV + ML)
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        app_ids      = applications_df["application_id"].tolist()
        all_features = compute_features_batch(app_ids)

        rows = []
        for app_id in app_ids:
            f = all_features.get(app_id, {})
            rows.append({feat: f.get(feat, 0) for feat in MODEL_FEATURES})

        # ── Single model call ──────────────────────────────
        features_df = pd.DataFrame(rows)[MODEL_FEATURES]
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

        risk_scores = model.predict_proba(features_df)[:, 1]

        high = medium = low = 0
        for score in risk_scores:
            tier = get_risk_tier(float(score))
            if tier == "High":   high   += 1
            elif tier == "Medium": medium += 1
            else:                  low    += 1

        return {
            "total_applications": len(app_ids),
            "high":   high,
            "medium": medium,
            "low":    low
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
