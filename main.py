# =========================================================
# CREDITSENTINEL FASTAPI - RENDER READY
# =========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import os
import math

from typing import List

from feature_engine import compute_features

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "creditsentinel_model_v1.pkl"
    )
)

print("✅ Real model loaded")
print(type(model))

# =========================================================
# LOAD CSV FILES
# =========================================================

bank_df = pd.read_csv(
    os.path.join(BASE_DIR, "bank_statements.csv")
)

bureau_df = pd.read_csv(
    os.path.join(BASE_DIR, "bureau_data.csv")
)

gst_df = pd.read_csv(
    os.path.join(BASE_DIR, "gst_filings.csv")
)

applications_df = pd.read_csv(
    os.path.join(BASE_DIR, "loan_applications.csv")
)

print("✅ CSV files loaded")

# =========================================================
# HELPER: SAFELY CONVERT VALUES FOR JSON
# =========================================================

def safe_float(val, default=0.0):
    """Convert to float, replacing NaN/Inf with default."""
    try:
        result = float(val)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0):
    """Convert to int safely."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def safe_str(val, default=""):
    """Convert to string, replacing NaN with default."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return default
        return str(val)
    except (TypeError, ValueError):
        return default


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Loan Risk Scoring API",
    description="API for predicting loan application risk tiers",
    version="1.0"
)

# =========================================================
# ENABLE CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST MODELS
# =========================================================

class ScoreRequest(BaseModel):
    application_id: str


class BatchScoreRequest(BaseModel):
    application_ids: List[str]


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "features": 43
    }


# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================

@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        # STEP 1: COMPUTE FEATURES
        features_dict = compute_features(req.application_id)

        print("\n================================")
        print("APPLICATION:", req.application_id)
        print("================================")

        # STEP 2: GET MODEL FEATURES
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_"):
            model_features = list(model.feature_name_)
        else:
            return {
                "application_id": req.application_id,
                "model_loaded": False,
                "error": "Model feature names not found"
            }

        print("Model expects:", len(model_features), "features")

        # STEP 3: FILTER FEATURES
        features_filtered = {
            feature: features_dict.get(feature, 0)
            for feature in model_features
        }

        # STEP 4: CREATE DATAFRAME
        features_df = pd.DataFrame([features_filtered])

        # STEP 5: CORRECT FEATURE ORDER
        features_df = features_df[model_features]

        # STEP 6: CLEAN VALUES (NaN, Inf → 0)
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # STEP 7: CONVERT NUMERIC
        for col in features_df.columns:
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)
            except Exception:
                pass

        # DEBUG
        print("\n========== FEATURES ==========")
        for col in features_df.columns[:10]:
            print(col, "=", features_df.iloc[0][col])
        print("==============================")

        # STEP 8: PREDICT
        prediction = model.predict_proba(features_df)
        print("\nPrediction Array:")
        print(prediction)

        risk_score = round(float(prediction[:, 1][0]), 4)

        # STEP 9: RISK TIER
        if risk_score < 0.4:
            tier = "Low"
        elif risk_score < 0.65:
            tier = "Medium"
        else:
            tier = "High"

        return {
            "application_id": req.application_id,
            "model_loaded": True,
            "risk_score": risk_score,
            "risk_tier": tier,
            "features_used": len(features_df.columns)
        }

    except Exception as e:
        return {
            "application_id": req.application_id,
            "model_loaded": False,
            "error": str(e)
        }


# =========================================================
# SCORE BATCH APPLICATIONS
# =========================================================

@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):

    results = []

    for app_id in req.application_ids:

        try:

            features_dict = compute_features(app_id)

            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
            else:
                model_features = list(model.feature_name_)

            features_filtered = {
                name: features_dict.get(name, 0)
                for name in model_features
            }

            features_df = pd.DataFrame([features_filtered])
            features_df = features_df[model_features]
            features_df = features_df.fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)

            for col in features_df.columns:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)
                except Exception:
                    pass

            risk_score = round(
                float(model.predict_proba(features_df)[:, 1][0]),
                4
            )

            if risk_score < 0.3:
                tier = "Low"
            elif risk_score < 0.6:
                tier = "Medium"
            else:
                tier = "High"

            results.append({
                "application_id": app_id,
                "model_loaded": True,
                "risk_score": risk_score,
                "risk_tier": tier,
                "features_used": len(features_filtered)
            })

        except Exception as e:
            results.append({
                "application_id": app_id,
                "model_loaded": False,
                "error": str(e)
            })

    return {
        "total_applications": len(req.application_ids),
        "results": results
    }


# =========================================================
# APPLICATION LIST ENDPOINT  ← FIXED
# =========================================================

@app.get("/api/applications")
def get_applications():

    applications = []

    for _, row in applications_df.iterrows():

        application = {
            "application_id":   safe_str(row.get("application_id", "")),
            "applicant_name":   safe_str(row.get("applicant_name", "")),
            "monthly_income":   safe_float(row.get("monthly_income", 0)),
            "loan_amount":      safe_float(row.get("loan_amount", 0)),
            "foir":             safe_float(row.get("foir", 0)),
            "risk_score":       safe_float(row.get("risk_score", 0)),
            "risk_tier":        safe_str(row.get("risk_tier", "Low")),
            "credit_score":     safe_int(row.get("cibil_score", 0)),
            "application_status": safe_str(row.get("application_status", "Pending")),
            "date_applied":     safe_str(row.get("date_applied", ""))
        }

        applications.append(application)

    return {
        "total": len(applications),
        "applications": applications
    }


# =========================================================
# SINGLE APPLICATION DETAIL ENDPOINT  ← FIXED
# =========================================================

@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

    matched = applications_df[
        applications_df["application_id"] == application_id
    ]

    if len(matched) == 0:
        return {"error": "Application not found"}

    row = matched.iloc[0]

    monthly_income = safe_float(row.get("monthly_income", 0))
    monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

    if monthly_income > 0:
        foir = round((monthly_emi / monthly_income) * 100, 2)
    else:
        foir = 0.0

    # GET LIVE SCORE
    score_result = score_application(
        ScoreRequest(application_id=application_id)
    )

    return {
        "application_id":     safe_str(row.get("application_id", "")),
        "applicant_name":     safe_str(row.get("applicant_name", "")),
        "monthly_income":     monthly_income,
        "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
        "foir":               foir,
        "risk_score":         safe_float(score_result.get("risk_score", 0)),
        "risk_tier":          safe_str(score_result.get("risk_tier", "Low")),
        "credit_score":       safe_int(row.get("cibil_score", 0)),
        "application_status": safe_str(row.get("application_status", "Pending")),
        "date_applied":       safe_str(row.get("date_applied", ""))
    }


# =========================================================
# PORTFOLIO SUMMARY
# =========================================================

@app.get("/api/portfolio/summary")
def portfolio_summary():

    approved = int((applications_df["application_status"] == "Approved").sum())
    rejected = int((applications_df["application_status"] == "Rejected").sum())
    pending  = int((applications_df["application_status"] == "Pending").sum())

    return {
        "total_applications": len(applications_df),
        "approved":  approved,
        "rejected":  rejected,
        "pending":   pending
}
