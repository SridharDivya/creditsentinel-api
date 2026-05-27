# =========================================================
# CREDITSENTINEL FASTAPI - RENDER DEPLOYMENT VERSION
# FULL WORKING CODE
# =========================================================

# =========================================================
# IMPORTS
# =========================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import traceback
import os

from typing import List

# =========================================================
# IMPORT FEATURE ENGINE
# =========================================================
from feature_engine import compute_features
from config import Config

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="CreditSentinel API",
    version="1.0"
)

# =========================================================
# ENABLE CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = os.getcwd()

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "creditsentinel_model_v1.pkl"
)

model = joblib.load(MODEL_PATH)

print("✅ Model Loaded")

# =========================================================
# LOAD APPLICATION DATA
# =========================================================
applications_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "loan_applications.csv"
    )
)

print("✅ Applications CSV Loaded")
print(applications_df.head())

# =========================================================
# REQUEST MODELS
# =========================================================
class ScoreRequest(BaseModel):

    application_id: str


class BatchScoreRequest(BaseModel):

    application_ids: List[str]

# =========================================================
# SAFE HELPERS
# =========================================================
def safe_float(value):

    try:

        if pd.isna(value):

            return 0.0

        return round(float(value), 2)

    except:

        return 0.0


def safe_int(value):

    try:

        if pd.isna(value):

            return 0

        return int(float(value))

    except:

        return 0


def safe_str(value):

    try:

        if pd.isna(value):

            return ""

        return str(value)

    except:

        return ""

# =========================================================
# HEALTH ENDPOINT
# =========================================================
@app.get("/health")
def health():

    return {

        "status": "ok",
        "model_loaded": True,
        "total_applications":
        len(applications_df)
    }

# =========================================================
# GENERATE REAL MODEL SCORE
# =========================================================
def generate_risk_score(application_id):

    try:

        # =========================================
        # COMPUTE FEATURES
        # =========================================
        features_dict = compute_features(
            application_id
        )

        # =========================================
        # MODEL FEATURE ORDER
        # =========================================
        if hasattr(model, "feature_names_in_"):

            model_features = list(
                model.feature_names_in_
            )

        else:

            model_features = list(
                model.feature_name_
            )

        # =========================================
        # FILTER FEATURES
        # =========================================
        filtered_features = {

            feature:
            features_dict.get(feature, 0)

            for feature in model_features
        }

        # =========================================
        # DATAFRAME
        # =========================================
        features_df = pd.DataFrame(
            [filtered_features]
        )

        features_df = features_df[
            model_features
        ]

        features_df = features_df.fillna(0)

        features_df = features_df.astype(float)

        # =========================================
        # PREDICT
        # =========================================
        prediction = model.predict_proba(
            features_df
        )

        risk_score = round(
            float(prediction[:,1][0]),
            4
        )

        # =========================================
        # RISK TIER
        # =========================================
        if risk_score < 0.4:

            risk_tier = "Low"

        elif risk_score < 0.65:

            risk_tier = "Medium"

        else:

            risk_tier = "High"

        return {

            "risk_score":
            risk_score,

            "risk_tier":
            risk_tier
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "risk_score": 0.0,
            "risk_tier": "Low"
        }

# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        result = generate_risk_score(
            req.application_id
        )

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            True,

            "risk_score":
            result["risk_score"],

            "risk_tier":
            result["risk_tier"],

            "features_used":
            43
        }

    except Exception as e:

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            False,

            "error":
            str(e)
        }

# =========================================================
# SCORE BATCH
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):

    results = []

    for application_id in req.application_ids:

        result = generate_risk_score(
            application_id
        )

        results.append({

            "application_id":
            application_id,

            "risk_score":
            result["risk_score"],

            "risk_tier":
            result["risk_tier"]
        })

    return {

        "total_applications":
        len(results),

        "results":
        results
    }

# =========================================================
# APPLICATIONS LIST ENDPOINT
# =========================================================
@app.get("/api/applications")
def get_applications():

    applications = []

    for _, row in applications_df.iterrows():

        app_id = safe_str(row.get("application_id", ""))

        score_result = score_application(
            ScoreRequest(application_id=app_id)
        )

        risk_score = safe_float(score_result.get("risk_score", 0))
        risk_tier  = safe_str(score_result.get("risk_tier", "Unknown"))

        credit_score = int(300 + (1 - risk_score) * 600)

        status_map = {
            "Low":    "Approved",
            "Medium": "Under Review",
            "High":   "Rejected"
        }
        application_status = status_map.get(risk_tier, "Pending")

        # ── FOIR Calculation ───────────────────────────────
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

        if monthly_income > 0:
            foir = round((monthly_emi / monthly_income) * 100, 2)
        else:
            foir = 0.0

        applications.append({
            "application_id":     app_id,
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "foir":               foir,               # ✅ Added below applicant
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
            "risk_score":         risk_score,
            "risk_tier":          risk_tier,
            "credit_score":       credit_score,
            "application_status": application_status
        })

    return {
        "total":        len(applications),
        "applications": applications
    }
# =========================================================
# APPLICATION DETAIL ENDPOINT
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

    try:

        if "application_id" not in applications_df.columns:
            return {"error": "application_id column missing"}

        matched = applications_df[
            applications_df["application_id"] == application_id
        ]

        if len(matched) == 0:
            return {"error": "Application not found"}

        row = matched.iloc[0]

        # ── FOIR Calculation ──────────────────────────────
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))
        foir = round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0.0

        # ── ML Score ──────────────────────────────────────
        score_result = score_application(
            ScoreRequest(application_id=application_id)
        )

        risk_score = safe_float(score_result.get("risk_score", 0))
        risk_tier  = safe_str(score_result.get("risk_tier", "Unknown"))

        # ── Credit Score from ML (same formula as list) ───
        credit_score = int(300 + (1 - risk_score) * 600)  # ✅ ML-derived

        # ── Status from risk tier ─────────────────────────
        status_map = {
            "Low":    "Approved",
            "Medium": "Under Review",
            "High":   "Rejected"
        }
        application_status = status_map.get(risk_tier, "Pending")

        return {
            "application_id":     safe_str(row.get("application_id", "")),
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
            "foir":               foir,
            "risk_score":         risk_score,
            "risk_tier":          risk_tier,
            "credit_score":       credit_score,        # ✅ ML-derived, not from CSV
            "application_status": application_status   # ✅ ML-derived, not from CSV
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    # Get all applications
    applications = db.query(Application).all()
    
    total = len(applications)
    
    # Count by RISK TIER (not status)
    high_count = 0
    medium_count = 0
    low_count = 0
    
    for app in applications:
        # Use risk_score to determine tier
        risk_score = app.risk_score if app.risk_score else 0
        
        # Define thresholds
        if risk_score >= 0.7:  # High risk
            high_count += 1
        elif risk_score >= 0.4:  # Medium risk
            medium_count += 1
        else:  # Low risk (0 to 0.4)
            low_count += 1
    
    return {
        "total_applications": total,
        "high": high_count,
        "medium": medium_count,
        "low": low_count
    }
