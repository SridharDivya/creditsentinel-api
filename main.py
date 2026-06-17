from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import traceback
import os
import math
import time
import json
from datetime import datetime

from typing import List, Optional

from feature_engine import compute_features
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# =========================================================
# DATABASE CONNECTION (RENDER POSTGRESQL)
# =========================================================
import psycopg2
from fastapi.responses import JSONResponse




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
# =========================================================
DB_CONFIG = {
    "host":     "dpg-d8j9bhernols73cff9t0-a",
    "port":     5432,
    "database": "creditsentinel_db_r67r",
    "user":     "creditsentinel_db_r67r_user",
    "password": "u3VnrUHo8cSzQlxgxYgfYQfOVV2fsupZ"
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
# =========================================================
def compute_cibil_score(row) -> int:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    foir               = safe_float(row_dict.get("foir", 0))
    monthly_income     = safe_float(row_dict.get("monthly_income", 0))
    loan_to_income     = safe_float(row_dict.get("loan_to_income_ratio", 0))
    num_existing_loans = safe_float(row_dict.get("num_existing_loans", 0))
    employment_years   = safe_float(row_dict.get("employment_years", 0))

    score = 750.0

    # FOIR
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

    # Monthly income
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

    # Loan-to-income ratio
    if loan_to_income <= 2:
        score += 30
    elif loan_to_income <= 4:
        score += 10
    elif loan_to_income <= 6:
        score -= 20
    else:
        score -= 50

    # Number of existing loans
    if num_existing_loans == 0:
        score += 20
    elif num_existing_loans == 1:
        score += 5
    elif num_existing_loans == 2:
        score -= 15
    else:
        score -= 30 * (num_existing_loans - 2)

    # Employment years
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

# FIX 1: Added analyst_name field to DecisionRequest
class DecisionRequest(BaseModel):
    decision:     str
    notes:        Optional[str] = ""
    analyst_name: Optional[str] = "Unknown"   # ← ADDED: was missing, caused analyst_name = null
    timestamp:    Optional[str] = None

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
    start_time = time.time()

    try:
        result = generate_risk_score(req.application_id)
        latency_ms = (time.time() - start_time) * 1000

        log_entry = {
            "timestamp":      datetime.now().isoformat(),
            "application_id": req.application_id,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"],
            "latency_ms":     round(latency_ms, 2),
            "status":         "success"
        }

        with open("model_predictions.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "application_id": req.application_id,
            "model_loaded":   True,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"],
            "features_used":  len(MODEL_FEATURES),
            "latency_ms":     round(latency_ms, 2)
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_entry = {
            "timestamp":      datetime.now().isoformat(),
            "application_id": req.application_id,
            "latency_ms":     round(latency_ms, 2),
            "status":         "error",
            "error":          str(e)
        }

        with open("model_predictions.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "application_id": req.application_id,
            "model_loaded":   False,
            "error":          str(e)
        }

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
                "cibil_score":        compute_cibil_score(row),
                "credit_score":       get_ml_credit_score(risk_score),
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
        cibil_score  = compute_cibil_score(row)
        credit_score = get_ml_credit_score(risk_score)

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
# =========================================================
@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # FIX 2: Corrected indentation (was 7 spaces, must be 8)
        query = """
SELECT audit_id,
       decision,
       decision_notes,
       timestamp,
       analyst_name
FROM audit_trail
WHERE application_id = %s
ORDER BY timestamp DESC
"""

        cursor.execute(query, (application_id,))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            return {"history": []}

        history = []
        for row in rows:
            # FIX 3: Corrected indentation on history.append block
            history.append({
                "audit_id":     row[0],
                "decision":     row[1],
                "notes":        row[2],
                "timestamp":    row[3].isoformat() if row[3] else None,
                "applicant_name": row[4]
            })

        return {"history": history}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# NOTIFICATION HELPERS
# =========================================================
def send_email(application_id: str, subject: str, body: str):
    try:
        recipient = f"{application_id.lower()}@example.com"

        message = MIMEMultipart()
        message["From"]    = FROM_EMAIL
        message["To"]      = recipient
        message["Subject"] = subject

        message.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(MAILTRAP_HOST, MAILTRAP_PORT)
        server.starttls()
        server.login(MAILTRAP_USERNAME, MAILTRAP_PASSWORD)
        server.sendmail(FROM_EMAIL, recipient, message.as_string())
        server.quit()

        print(f"Mailtrap email sent to {recipient}")
        return True

    except Exception as e:
        print(f"Mailtrap email failed: {str(e)}")
        return False


def notify_team_lead(application_id: str):
    print("\n===== INTERNAL NOTIFICATION =====")
    print(f"{application_id} assigned to TEAM_LEAD")
    print("=================================\n")
    return True

# =========================================================
# AUDIT TRAIL — POST /api/applications/{id}/process-decision
# =========================================================
@app.post("/api/applications/{application_id}/process-decision")
def process_decision(application_id: str, req: DecisionRequest):

    valid_decisions = ["APPROVE", "REJECT", "REVIEW"]

    if req.decision.upper() not in valid_decisions:
        return {"status": "failed", "error": "Invalid decision"}

    notes = req.notes or ""
    conn  = None

    # ✅ FIX: backend-controlled analyst identity
    # GET REAL APPLICANT NAME FROM CSV (fallback safe)
matched = applications_df[
    applications_df["application_id"].astype(str) == str(application_id)
]

applicant_name = (
    safe_str(matched.iloc[0]["applicant_name"])
    if len(matched) > 0
    else "Unknown Applicant"
)

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        notification_sent = False
        notification_type = None

        # APPROVE
        if req.decision.upper() == "APPROVE":

            cursor.execute("""
                UPDATE applications
                SET application_status = 'approved',
                    updated_at = CURRENT_TIMESTAMP
                WHERE application_id = %s
            """, (application_id,))

            notification_sent = send_email(
                application_id,
                "Loan Approved",
                "Congratulations! Your loan application has been approved."
            )
            notification_type = "approval_email"

        # REJECT
        elif req.decision.upper() == "REJECT":

            cursor.execute("""
                UPDATE applications
                SET application_status = 'rejected',
                    updated_at = CURRENT_TIMESTAMP
                WHERE application_id = %s
            """, (application_id,))

            notification_sent = send_email(
                application_id,
                "Loan Rejected",
                f"Your loan application was rejected.\nReason: {notes}"
            )
            notification_type = "rejection_email"

        # REVIEW
        elif req.decision.upper() == "REVIEW":

            cursor.execute("""
                UPDATE applications
                SET application_status = 'under_review',
                    assigned_reviewer = 'TEAM_LEAD',
                    updated_at = CURRENT_TIMESTAMP
                WHERE application_id = %s
            """, (application_id,))

            notification_sent = notify_team_lead(application_id)
            notification_type = "internal_review_notification"

        # AUDIT TRAIL (FIXED)
        cursor.execute("""
    INSERT INTO audit_trail
    (application_id, decision, decision_notes, analyst_name, timestamp)
    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
    RETURNING audit_id
""", (
    application_id,
    req.decision.upper(),
    notes,
    applicant_name   # ✅ REPLACED
))
        audit_id = cursor.fetchone()[0]

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "application_id": application_id,
            "audit_id": audit_id,
            "status": req.decision.lower(),
            "next_action": notification_type,
            "notification_sent": notification_sent,
            "message": "Decision processed successfully",
            "applicant_name": applicant_name   # ✅ FIXED OUTPUT
        }

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()

        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "application_id": application_id,
                "error": str(e)
            }
        )
# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    start = time.time()
    try:
        df = applications_df

        def get_col(name):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(0)
            return pd.Series(0.0, index=df.index)

        monthly_income     = get_col("monthly_income")
        num_existing_loans = get_col("num_existing_loans")
        employment_years   = get_col("employment_years")
        foir               = get_col("foir")
        loan_to_income     = get_col("loan_to_income_ratio")

        score = pd.Series(750.0, index=df.index)

        # FOIR
        score += np.select(
            [foir <= 30, foir <= 40, foir <= 50, foir <= 60],
            [40, 10, -20, -60],
            default=-100
        )

        # Monthly income
        score += np.select(
            [monthly_income >= 100000, monthly_income >= 75000,
             monthly_income >= 50000,  monthly_income >= 30000],
            [50, 35, 20, 5],
            default=-20
        )

        # Loan-to-income
        score += np.select(
            [loan_to_income <= 2, loan_to_income <= 4, loan_to_income <= 6],
            [30, 10, -20],
            default=-50
        )

        # Existing loans
        extra_penalty = np.where(num_existing_loans > 2, -30 * (num_existing_loans - 2), 0)
        score += np.select(
            [num_existing_loans == 0, num_existing_loans == 1, num_existing_loans == 2],
            [20, 5, -15],
            default=extra_penalty
        )

        # Employment years
        score += np.select(
            [employment_years >= 10, employment_years >= 5,
             employment_years >= 3,  employment_years >= 1],
            [40, 25, 10, -5],
            default=-25
        )

        score = score.clip(300, 900).astype(int)

        low    = int((score >= 750).sum())
        medium = int(((score >= 650) & (score < 750)).sum())
        high   = int((score < 650).sum())

        elapsed = round(time.time() - start, 2)
        print(f"✅ Portfolio Summary: high={high}, medium={medium}, low={low}, time={elapsed}s")

        return {
            "total_applications":     TOTAL_APPLICATIONS,
            "high":                   high,
            "medium":                 medium,
            "low":                    low,
            "execution_time_seconds": elapsed
        }

    except Exception as e:
        err = traceback.format_exc()
        print("PORTFOLIO ERROR:", err)
        return {"error": str(e), "detail": err}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
   
