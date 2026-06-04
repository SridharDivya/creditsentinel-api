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

model = joblib.load(os.path.join(BASE_DIR, "lightgbm_0.8106.pkl"))
print("✅ Model Loaded")

applications_df = pd.read_csv(os.path.join(BASE_DIR, "loan_applications.csv"))
print(f"✅ Applications Loaded: {len(applications_df)} rows")

# =========================================================
# DEBUG: Print available columns once at startup
# =========================================================
print(f"✅ CSV Columns: {list(applications_df.columns)}")

# =========================================================
# MODEL FEATURES (resolved once at startup)
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
    """Derive a realistic credit score from the ML risk score (300–900 range)."""
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
# CREDIT SCORE RESOLVER
#
# TWO SEPARATE FUNCTIONS — fixes the root cause of the bug:
#
#   cibil_score  = raw bureau score read directly from the CSV row
#                  Returns 0 if the column is missing or value is invalid.
#                  This is what the Application Detail page should display.
#
#   credit_score = ML-derived score calculated from the risk_score
#                  Always returns a value in the 300–900 range.
#                  This is what the Risk Score service uses internally.
#
# Previously both fields called resolve_credit_score() which returned
# the same value, causing the "same data for both" symptom you reported.
# =========================================================

# All possible column names your CSV might use — add more if needed
CIBIL_COLUMN_CANDIDATES = [
    "cibil_score", "credit_score", "cibil", "bureau_score",
    "creditScore", "CIBIL Score", "CIBIL_score", "Credit Score",
    "bureau_credit_score", "score", "fico_score", "credit_rating"
]

# Resolved once at startup — find which column actually exists in the CSV
CIBIL_COLUMN = None
for _col in CIBIL_COLUMN_CANDIDATES:
    if _col in applications_df.columns:
        CIBIL_COLUMN = _col
        print(f"✅ CIBIL/bureau score column found: '{CIBIL_COLUMN}'")
        break

if CIBIL_COLUMN is None:
    print("⚠️  No CIBIL/bureau score column found in CSV — cibil_score will be 0.")


def get_raw_cibil_score(row) -> int:
    """
    Reads the bureau/CIBIL score DIRECTLY from the CSV row.

    - Returns the integer score if the column exists and the value is valid.
    - Returns 0 if the column is missing, null, or non-numeric.
    - Does NOT fall back to any ML-derived value — keeps the two fields distinct.

    FIX: Previously both cibil_score and credit_score called resolve_credit_score(),
    which silently replaced a missing/zero CIBIL value with the ML-derived score,
    making both fields identical and hiding the real data gap.
    """
    if CIBIL_COLUMN is None:
        return 0  # Column doesn't exist in CSV at all

    # ── FIX: pandas Series.get() does NOT work like dict.get() ──────────────
    # row.get("col", None) on a pandas Series will raise an error or return
    # unexpected results when the column is missing.  Convert to dict first.
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    val = row_dict.get(CIBIL_COLUMN, None)

    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0  # Null / NaN in CSV

    try:
        score = int(float(val))
        return score if score > 0 else 0   # treat 0 or negative as missing
    except (ValueError, TypeError):
        return 0  # Non-numeric value in CSV


def get_ml_credit_score(risk_score: float) -> int:
    """
    Returns an ML-derived credit score (300–900) from the model's risk score.
    This is the score the Risk Score service has always computed correctly.
    """
    return get_credit_score_from_risk(risk_score)


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
        "total_applications": len(applications_df),
        "cibil_score_column": CIBIL_COLUMN or "not_found_in_csv"
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

            # ── FIX: cibil_score and credit_score are now DIFFERENT values ──
            # cibil_score  = raw bureau score from the CSV row
            # credit_score = ML-derived score (300–900) from risk model
            applications.append({
                "application_id":     app_id,
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         risk_score,
                "risk_tier":          risk_tier,
                "cibil_score":        get_raw_cibil_score(row),        # ✅ raw CSV bureau score
                "credit_score":       get_ml_credit_score(risk_score), # ✅ ML-derived 300–900
                "application_status": get_status(risk_tier)
            })

        return {
            "total":        len(applications_df),
            "applications": applications
        }

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

        row = matched.iloc[0]

        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

        foir = (
            round((monthly_emi / monthly_income) * 100, 2)
            if monthly_income > 0 else 0
        )

        score_data = generate_risk_score(application_id)
        risk_score = score_data["risk_score"]
        risk_tier  = score_data["risk_tier"]

        # ── FIX: read each score from its correct, dedicated source ─────────
        # Before this fix, both fields called resolve_credit_score() which
        # silently replaced a missing/zero CIBIL value with the ML-derived
        # score, so both fields always showed the same number.
        cibil_score  = get_raw_cibil_score(row)        # raw CSV bureau score (e.g. 573, 706)
        credit_score = get_ml_credit_score(risk_score) # ML-derived score (300–900)

        application_status = safe_str(
            row.get("application_status", row.get("status", ""))
        )
        if application_status == "":
            if risk_score >= 0.75:
                application_status = "Rejected"
            elif risk_score >= 0.45:
                application_status = "Pending"
            else:
                application_status = "Approved"

        print(
            f"Application={application_id} | "
            f"CIBIL(raw)={cibil_score} | "
            f"CreditScore(ML)={credit_score} | "
            f"Risk={risk_score}"
        )

        return {
            "application_id":     safe_str(row.get("application_id", "")),
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", row.get("loan_amount", 0))),
            "foir":               foir,
            "cibil_score":        cibil_score,         # ✅ real bureau score from CSV
            "credit_score":       credit_score,        # ✅ ML-derived score, always 300–900
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

        total       = len(applications_df)
        sample_size = len(sample_df)
        scale       = total / sample_size

        return {
            "total_applications": total,
            "high":   round(high   * scale),
            "medium": round(medium * scale),
            "low":    round(low    * scale)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
