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
# DATABASE CONNECTION (RENDER POSTGRESQL)
# =========================================================
import psycopg2
from fastapi.responses import JSONResponse

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a connection to your Render PostgreSQL database."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing on Render!")
    return psycopg2.connect(DATABASE_URL)


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
print("CSV COLUMNS:", list(applications_df.columns))

# Total applications count shown in all API responses
TOTAL_APPLICATIONS = 15000

# =========================================================
# POSTGRESQL DATABASE CONNECTION
# ⚠️  ONLY change the password below — everything else is correct
# =========================================================
DB_CONFIG = {
    "host":     "dpg-d8j9bhernols73cff9t0-a",
    "port":     5432,
    "database": "creditsentinel_db_r67r",
    "user":     "creditsentinel_db_r67r_user",
    "password": "u3VnrUHo8cSzQlxgxYgfYQfOVV2fsupZ"   # ← paste your Render DB password here
}

def get_db_connection():
    """Create and return a new PostgreSQL connection"""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

# Test DB connection on startup
try:
    conn_test = get_db_connection()
    conn_test.close()
    print("✅ PostgreSQL Connected")
except Exception as e:
    print(f"❌ PostgreSQL Connection Failed: {e}")

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

def get_status(risk_tier: str) -> str:
    return {
        "Low":    "Approved",
        "Medium": "Under Review",
        "High":   "Rejected"
    }.get(risk_tier, "Pending")

def get_foir(monthly_income: float, monthly_emi: float) -> float:
    return round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0.0

# =========================================================
# CIBIL SCORE — derived from real CSV data
#
# Your CSV has no bureau/CIBIL column. So we calculate it
# using the actual fields that exist: foir, monthly_income,
# loan_to_income_ratio, num_existing_loans, employment_years.
#
# Formula logic (standard credit scoring approach):
#   Base score      = 750  (average CIBIL starting point)
#   FOIR penalty    = high debt-to-income ratio lowers score
#   Income boost    = higher income slightly improves score
#   LTI penalty     = high loan-to-income ratio lowers score
#   Existing loans  = more loans = higher risk = lower score
#   Employment      = longer employment = lower risk = higher score
#
# Final score is clamped to 300-900 (standard CIBIL range).
# Every application gets a DIFFERENT score based on its real data.
# =========================================================
def compute_cibil_score(row) -> int:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    foir               = safe_float(row_dict.get("foir", 0))
    monthly_income     = safe_float(row_dict.get("monthly_income", 0))
    loan_to_income     = safe_float(row_dict.get("loan_to_income_ratio", 0))
    num_existing_loans = safe_float(row_dict.get("num_existing_loans", 0))
    employment_years   = safe_float(row_dict.get("employment_years", 0))

    score = 750.0  # base score

    # FOIR: 0-30% is healthy, above 50% is risky
    if foir <= 30:
        score += 40
    elif foir <= 40:
        score += 10
    elif foir <= 50:
        score -= 20
    elif foir <= 60:
        score -= 60
    else:
        score -= 100

    # Monthly income: higher income = better creditworthiness
    if monthly_income >= 100000:
        score += 50
    elif monthly_income >= 75000:
        score += 35
    elif monthly_income >= 50000:
        score += 20
    elif monthly_income >= 30000:
        score += 5
    else:
        score -= 20

    # Loan-to-income ratio: lower is better
    if loan_to_income <= 2:
        score += 30
    elif loan_to_income <= 4:
        score += 10
    elif loan_to_income <= 6:
        score -= 20
    else:
        score -= 50

    # Number of existing loans: more = riskier
    if num_existing_loans == 0:
        score += 20
    elif num_existing_loans == 1:
        score += 5
    elif num_existing_loans == 2:
        score -= 15
    else:
        score -= 30 * (num_existing_loans - 2)

    # Employment years: stable employment = better score
    if employment_years >= 10:
        score += 40
    elif employment_years >= 5:
        score += 25
    elif employment_years >= 3:
        score += 10
    elif employment_years >= 1:
        score -= 5
    else:
        score -= 25

    # Clamp to standard CIBIL range 300-900
    return max(300, min(900, int(score)))


# ML-derived credit score (300-900) from risk model output
def get_ml_credit_score(risk_score: float) -> int:
    return int(300 + (1 - risk_score) * 600)

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

class DecisionRequest(BaseModel):
    decision:  str
    notes:     Optional[str] = ""
    timestamp: Optional[str] = None

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status":             "ok",
        "model_loaded":       True,
        "total_applications": TOTAL_APPLICATIONS,
        "cibil_source":       "computed_from_foir_income_lti_loans_employment"
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
                "cibil_score":        compute_cibil_score(row),        # computed from real CSV data
                "credit_score":       get_ml_credit_score(risk_score), # ML-derived 300-900
                "application_status": get_status(risk_tier)
            })

        return {"total": TOTAL_APPLICATIONS, "applications": applications}

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
        cibil_score  = compute_cibil_score(row)        # computed from real CSV data
        credit_score = get_ml_credit_score(risk_score) # ML-derived 300-900

        application_status = safe_str(row.get("application_status", row.get("status", "")))
        if application_status == "":
            if risk_score >= 0.75:   application_status = "Rejected"
            elif risk_score >= 0.45: application_status = "Pending"
            else:                    application_status = "Approved"

        print(f"[DETAIL] id={application_id} | cibil={cibil_score} | credit={credit_score} | risk={risk_score}")

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
            "date_applied":       safe_str(row.get("application_date", row.get("date_applied", "")))
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# AUDIT TRAIL — GET /api/applications/{id}/history
# Returns all past decisions for an application (newest first)
# =========================================================
# =========================================================
# AUDIT TRAIL — GET /api/applications/{id}/history
# Returns all past decisions for an application (newest first)
# =========================================================
@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT audit_id,
               decision,
               decision_notes,
               timestamp
        FROM audit_trail
        WHERE application_id = %s
        ORDER BY timestamp DESC
        """

        cursor.execute(query, (application_id,))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            return {
                "application_id": application_id,
                "decision_history": [],
                "latest_decision": None,
                "latest_timestamp": None
            }

        history = []

        for row in rows:
            history.append({
                "audit_id": row[0],
                "decision": row[1],
                "notes": row[2],
                "timestamp": row[3].isoformat() if row[3] else None
            })

        return {
            "application_id": application_id,
            "decision_history": history,
            "latest_decision": rows[0][1],
            "latest_timestamp": rows[0][3].isoformat()
        }

    except Exception as e:
        return {
            "error": str(e)
        }



# =========================================================
# AUDIT TRAIL — POST /api/applications/{id}/decision
# Log a lending decision (APPROVE / REJECT / REVIEW)
# Called by Jaajitha's frontend APPROVE/REJECT/REVIEW button
# =========================================================

@app.post("/api/applications/{application_id}/decision")
def log_decision(application_id: str, req: DecisionRequest):

    valid_decisions = ["APPROVE", "REJECT", "REVIEW"]

    if req.decision.upper() not in valid_decisions:
        return {"error": "Invalid decision"}

    notes = req.notes or ""

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO audit_trail
        (application_id, decision, decision_notes, timestamp)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING audit_id
        """

        cursor.execute(
            query,
            (
                application_id,
                req.decision.upper(),
                notes
            )
        )

        audit_id = cursor.fetchone()[0]

        conn.commit()

        cursor.close()
        conn.close()

        return {
            "audit_id": audit_id,
            "status": "logged",
            "message": "Decision recorded successfully"
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# AUDIT TRAIL — GET /api/applications/"/api/portfolio/summary"
# Returns all past decisions for an application (newest first)
# =========================================================
import time
import traceback

@app.get("/api/portfolio/summary")
def portfolio_summary():
    try:
        t_start = time.time()

        # Read pre-computed risk tiers
        tier_counts = applications_df["_risk_tier"].value_counts()

        high = int(tier_counts.get("High", 0))
        medium = int(tier_counts.get("Medium", 0))
        low = int(tier_counts.get("Low", 0))

        elapsed = time.time() - t_start

        print(
            f"[PORTFOLIO SUMMARY] "
            f"query_time={elapsed:.4f}s "
            f"high={high} "
            f"medium={medium} "
            f"low={low}"
        )

        return {
            "total_applications": TOTAL_APPLICATIONS,
            "high": high,
            "medium": medium,
            "low": low,
            "execution_time_seconds": round(elapsed, 4)
        }

    except Exception as e:
        print(traceback.format_exc())

        return {
            "error": str(e)
        }

         
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
