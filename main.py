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

from feature_engine import compute_features  # ✅ only this

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


model=joblib.load(os.path.join(BASE_DIR,"lightgbm_0.8106.pkl"))  # ✅ Using Config
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
# APPLICATIONS LIST
# =========================================================
@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    try:
        applications = []

        subset = applications_df.iloc[offset: offset + limit]

        for _, row in subset.iterrows():
            app_id = safe_str(row.get("application_id", ""))

            result     = generate_risk_score(app_id)
            risk_score = result["risk_score"]
            risk_tier  = result["risk_tier"]

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
                "credit_score":       get_credit_score(risk_score),
                "application_status": get_status(risk_tier)
            })

        return {
            "total":        len(applications_df),  # always 15000
            "applications": applications
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}




@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

    try:

        matched = applications_df[
            applications_df["application_id"].astype(str)
            == str(application_id)
        ]

        if len(matched) == 0:
            return {
                "error": "Application not found"
            }

        row = matched.iloc[0]

        # =====================================================
        # Monthly Income
        # =====================================================
        monthly_income = safe_float(
            row.get("monthly_income", 0)
        )

        # =====================================================
        # EMI
        # =====================================================
        monthly_emi = safe_float(
            row.get("existing_monthly_emi", 0)
        )

        # =====================================================
        # FOIR
        # =====================================================
        foir = (
            round((monthly_emi / monthly_income) * 100, 2)
            if monthly_income > 0
            else 0
        )

        # =====================================================
        # Risk Score from Model
        # =====================================================
        score_data = generate_risk_score(
            application_id
        )

        risk_score = score_data["risk_score"]
        risk_tier = score_data["risk_tier"]

        # =====================================================
        # CREDIT SCORE (ROBUST VERSION)
        # =====================================================
        credit_score = None

        possible_columns = [
            "cibil_score",
            "credit_score",
            "cibil",
            "bureau_score",
            "creditScore",
            "CIBIL Score"
        ]

        for col in possible_columns:

            if col in row.index:

                value = row[col]

                if pd.notna(value):

                    try:
                        credit_score = int(float(value))
                        break

                    except:
                        pass

        if credit_score is None:
            credit_score = 0

        # =====================================================
        # Application Status
        # =====================================================
        application_status = safe_str(
            row.get(
                "application_status",
                row.get("status", "")
            )
        )

        if application_status == "":

            if risk_score >= 0.75:
                application_status = "Rejected"

            elif risk_score >= 0.45:
                application_status = "Pending"

            else:
                application_status = "Approved"

        # =====================================================
        # DEBUG LOGS (optional)
        # =====================================================
        print(
            f"Application={application_id}, "
            f"Credit Score={credit_score}"
        )

        # =====================================================
        # RESPONSE
        # =====================================================
        return {

            "application_id": safe_str(
                row.get("application_id", "")
            ),

            "applicant_name": safe_str(
                row.get("applicant_name", "")
            ),

            "monthly_income": monthly_income,

            "loan_amount": safe_float(
                row.get(
                    "requested_loan_amount",
                    row.get("loan_amount", 0)
                )
            ),

            "foir": foir,

            "credit_score": credit_score,

            "risk_score": risk_score,

            "risk_tier": risk_tier,

            "application_status": application_status,

            "date_applied": safe_str(
                row.get(
                    "date_applied",
                    row.get("created_at", "")
                )
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        high = medium = low = 0

        # ── Sample 500 instead of 15,000 ──────────────────
        sample_df = applications_df.sample(
            n=min(500, len(applications_df)),
            random_state=42
        )

        for _, row in sample_df.iterrows():
            app_id = safe_str(row.get("application_id", ""))
            result = generate_risk_score(app_id)
            tier   = result["risk_tier"]

            if tier == "High":     high   += 1
            elif tier == "Medium": medium += 1
            else:                  low    += 1

        # ── Scale up to full 15,000 ───────────────────────
        total      = len(applications_df)
        sample_size = len(sample_df)

        scale = total / sample_size

        return {
            "total_applications": total,
            "high":   round(high   * scale),
            "medium": round(medium * scale),
            "low":    round(low    * scale)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
