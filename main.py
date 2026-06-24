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
import asyncio
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from typing import List, Optional

from feature_engine import compute_features
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# =========================================================
# DATABASE CONNECTION (RENDER POSTGRESQL)
# =========================================================
import psycopg2
from psycopg2 import pool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
print("APPLICATION COLUMNS:")
print(applications_df.columns.tolist())
# Total applications count shown in all API responses
TOTAL_APPLICATIONS = 15000

# =========================================================
# POSTGRESQL DATABASE CONNECTION WITH POOLING
# =========================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}
MAIL_HOST = os.getenv("MAIL_HOST")
MAIL_PORT = int(os.getenv("MAIL_PORT", 2525))
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM", "noreply@creditsentinel.com")
# Increased pool size to better handle concurrent requests
db_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=30,
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    database=DB_CONFIG["database"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"]
)

print("✅ Connection Pool Initialized")

# =========================================================
# ASYNC AUDIT LOGGING QUEUE
# Offloads DB audit writes to a background thread so the
# request response is not blocked by the INSERT latency.
# =========================================================
_audit_queue: asyncio.Queue = None
_audit_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audit")

def _audit_worker(payload: dict):
    """Blocking DB insert — runs in thread pool, not on the event loop."""
    conn = None

    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO audit_trail
            (
                application_id,
                decision,
                decision_notes,
                applicant_name,
                analyst_name,
                timestamp
            )
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING audit_id
        """, (
            payload["application_id"],
            payload["decision"],
            payload["notes"],
            payload["applicant_name"],
            payload["analyst_name"]
        ))

        audit_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return audit_id

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[AUDIT WORKER ERROR] {e}")
        return None

    finally:
        if conn:
            db_pool.putconn(conn)
async def fire_and_forget_audit(payload: dict):
    """Submit audit insert to background thread, don't block the response."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_audit_executor, _audit_worker, payload)

def get_db_connection():
    """Get connection from pool"""
    return db_pool.getconn()

try:
    conn_test = get_db_connection()
    db_pool.putconn(conn_test)
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

def send_email(recipient, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = MAIL_FROM
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(MAIL_HOST, MAIL_PORT)
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)

        server.sendmail(
            MAIL_FROM,
            recipient,
            msg.as_string()
        )

        server.quit()
        return True

    except Exception as e:
        print(f"EMAIL ERROR: {e}")
        return False

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
# FEATURE CACHE
# Caches compute_features() results per application_id so
# repeated calls (e.g. list + detail) don't recompute.
# TTL-style eviction is done via a simple size-bounded dict.
# =========================================================
_feature_cache: dict = {}
_feature_cache_lock = threading.Lock()
FEATURE_CACHE_MAX = 500  # keep most-recently-used IDs

def _get_cached_features(application_id: str) -> dict:
    with _feature_cache_lock:
        if application_id in _feature_cache:
            return _feature_cache[application_id]
    features = compute_features(application_id)
    with _feature_cache_lock:
        if len(_feature_cache) >= FEATURE_CACHE_MAX:
            # evict oldest entry
            oldest = next(iter(_feature_cache))
            del _feature_cache[oldest]
        _feature_cache[application_id] = features
    return features

# =========================================================
# CORE: RUN ML MODEL
# =========================================================
def generate_risk_score(application_id: str) -> dict:
    try:
        features_dict     = _get_cached_features(application_id)
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
# Parallelised with ThreadPoolExecutor to cut wall-clock
# time proportional to the number of IDs in the request.
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    with ThreadPoolExecutor(max_workers=min(8, len(req.application_ids))) as ex:
        futures = {ex.submit(generate_risk_score, app_id): app_id
                   for app_id in req.application_ids}
        results = []
        for future, app_id in futures.items():
            result = future.result()
            results.append({
                "application_id": app_id,
                "risk_score":     result["risk_score"],
                "risk_tier":      result["risk_tier"]
            })
    return {"total_applications": len(results), "results": results}

# =========================================================
# APPLICATIONS LIST
# UPDATED: added created_at and decision_date for Jaajitha
# =========================================================
@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    try:
        rows_list = []
        for i in range(offset, offset + limit):
            row = applications_df.iloc[i % len(applications_df)].copy()
            row["application_id"] = f"APP-{i+1:06d}"
            rows_list.append(row)
        subset = pd.DataFrame(rows_list)

        # Score all rows in parallel
        app_ids = [safe_str(row.get("application_id", "")) for _, row in subset.iterrows()]
        with ThreadPoolExecutor(max_workers=min(8, len(app_ids))) as ex:
            score_map = {
                app_id: future.result()
                for app_id, future in
                ((app_id, ex.submit(generate_risk_score, app_id)) for app_id in app_ids)
            }

        applications = []
        for _, row in subset.iterrows():
            app_id         = safe_str(row.get("application_id", ""))
            result         = score_map[app_id]
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
                "application_status": get_status(risk_tier),
                # --- NEW: date fields for Jaajitha's dashboard ---
                "created_at":         safe_str(row.get("created_at", row.get("application_date", ""))),
                "decision_date":      safe_str(row.get("decision_date", "")),
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
            try:
                numeric = int(str(application_id).split("-")[-1]) - 1
                row = applications_df.iloc[numeric % len(applications_df)].copy()
                row["application_id"] = application_id
            except Exception:
                return {"error": "Application not found"}
        else:
            row = matched.iloc[0]

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
        clean_search_id = str(application_id).strip().upper()

        matched = applications_df[
            applications_df["application_id"].astype(str).str.strip().str.upper() == clean_search_id
        ]

        if len(matched) == 0:
            try:
                numeric = int(clean_search_id.split("-")[-1]) - 1
                csv_applicant_name = safe_str(applications_df.iloc[numeric % len(applications_df)]["applicant_name"])
            except Exception:
                csv_applicant_name = "Unknown Applicant"
        else:
            csv_applicant_name = safe_str(matched.iloc[0]["applicant_name"])

        conn = get_db_connection()
        cursor = conn.cursor()

        # NOTE: ensure this index exists for fast lookups:
        #   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_trail_app_id
        #   ON audit_trail (UPPER(TRIM(application_id)));
        cursor.execute("""
            SELECT audit_id,
                   decision,
                   decision_notes,
                   timestamp,
                   applicant_name,
                   analyst_name
            FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC
        """, (clean_search_id,))
        rows = cursor.fetchall()

        cursor.close()
        db_pool.putconn(conn)

        if not rows:
            try:
                numeric = int(clean_search_id.split("-")[-1]) - 1
                csv_row = applications_df.iloc[numeric % len(applications_df)].copy()
                csv_row["application_id"] = application_id

                score_data  = generate_risk_score(application_id)
                risk_tier   = score_data["risk_tier"]
                decision    = {"Low": "APPROVE", "Medium": "REVIEW", "High": "REJECT"}.get(risk_tier, "REVIEW")
                status_note = f"Credit profile {risk_tier.lower()} risk — auto decision based on model score {score_data['risk_score']}"
                app_date    = safe_str(csv_row.get("application_date", ""))
                payload = {
                    "application_id": application_id,
                    "decision": decision,
                    "notes": status_note,
                    "applicant_name": csv_applicant_name,
                    "analyst_name": "SYSTEM"
                }
                

                 
                real_audit_id = _audit_worker(payload)

                return {
                    "history": [{
                        "audit_id":       real_audit_id,
                        "decision":       decision,
                        "notes":          status_note,
                        "timestamp":      app_date,
                        "applicant_name": csv_applicant_name,
                        "analyst_name":   ""
                    }]
                }

            except Exception as insert_err:
                print(f"[AUDIT AUTO-INSERT ERROR] {insert_err}")
                return {"history": []}

        history = []
        for row in rows:
            db_value = row[4]
            db_str   = str(db_value).strip() if db_value is not None else ""

            if db_value is None or db_str == "" or db_str.lower() in ["none", "null"]:
                final_applicant_name = csv_applicant_name
            else:
                final_applicant_name = safe_str(db_value)

            history.append({
                "audit_id":       row[0],
                "decision":       row[1],
                "notes":          row[2],
                "timestamp":      row[3].isoformat() if row[3] else None,
                "applicant_name": final_applicant_name,
                "analyst_name":   safe_str(row[5]) if len(row) > 5 else ""
            })

        return {"history": history}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# AUDIT TRAIL — POST /api/applications/{id}/process-decision
# UPDATED: tracks processing_time (decision duration in seconds)
# =========================================================
@app.post("/api/applications/{application_id}/process-decision")
async def process_decision(application_id: str, req: DecisionRequest):

    # NEW: start timer to track total decision processing time
    decision_start = time.time()

    decision_map = {
        "APPROVE":  "APPROVE",
        "APPROVED": "APPROVE",
        "REJECT":   "REJECT",
        "REJECTED": "REJECT",
        "REVIEW":   "REVIEW"
    }

    decision = str(req.decision).strip().upper()

    if decision not in decision_map:
        return {
            "status": "failed",
            "error": "Invalid decision. Allowed values: APPROVE, APPROVED, REJECT, REJECTED, REVIEW"
        }

    decision = decision_map[decision]
    notes = req.notes or ""
    conn = None

    search_id = str(application_id).strip().upper()

    matched = applications_df[
        applications_df["application_id"]
        .astype(str)
        .str.strip()
        .str.upper() == search_id
    ]

    if len(matched) == 0:
        try:
            numeric = int(search_id.split("-")[-1]) - 1
            matched = applications_df.iloc[[numeric % len(applications_df)]].copy()
            matched["application_id"] = application_id
        except Exception:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "failed",
                    "error": f"Application ID {application_id} not found"
                }
            )

    real_applicant_name = safe_str(
        matched.iloc[0].get("applicant_name", "Unknown Applicant")
    )
    recipient_email = safe_str(
        matched.iloc[0].get("email", "")
    )

    notification_sent = False
    notification_type = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if decision == "APPROVE":
            cursor.execute("""
                UPDATE applications
                SET application_status = 'approved',
                    updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "approval_email"

        elif decision == "REJECT":
            cursor.execute("""
                UPDATE applications
                SET application_status = 'rejected',
                    updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "rejection_email"

        elif decision == "REVIEW":
            cursor.execute("""
                UPDATE applications
                SET application_status = 'under_review',
                    assigned_reviewer = 'TEAM_LEAD',
                    updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "internal_review_notification"

        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        conn = None

        # ==========================
        # SEND EMAIL VIA MAILTRAP
        # ==========================

        if recipient_email:

            if decision == "APPROVE":
                send_email(
                    recipient_email,
                    "Loan Application Approved",
                    f"""
Hello {real_applicant_name},

Congratulations!

Your loan application {application_id} has been APPROVED.

Regards,
CreditSentinel Team
"""
                )

            elif decision == "REJECT":
                send_email(
                    recipient_email,
                    "Loan Application Rejected",
                    f"""
Hello {real_applicant_name},

Your loan application {application_id} has been REJECTED.

Reason:
{notes}

Regards,
CreditSentinel Team
"""
                )

            elif decision == "REVIEW":
                send_email(
                    recipient_email,
                    "Application Under Review",
                    f"""
Hello {real_applicant_name},

Your loan application {application_id} is currently UNDER REVIEW.

Our team will contact you shortly.

Regards,
CreditSentinel Team
"""
                )

    except Exception as e:
        if conn:
            conn.rollback()
            try:
                cursor.close()
            except Exception:
                pass
            db_pool.putconn(conn)

        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "application_id": application_id,
                "error": str(e)
            }
        )

     current_user_name = "Divya"

    audit_payload = {
        "application_id": application_id,
        "decision": decision,
        "notes": notes,
        "applicant_name": real_applicant_name,
        "analyst_name": current_user_name
    }

    audit_task = asyncio.create_task(
        fire_and_forget_audit(audit_payload)
    )

    audit_id = await audit_task
    

    # NEW: total time from start of function to response
    processing_time = round(time.time() - decision_start, 3)

    return {
        "application_id":    application_id,
        "applicant_name":    real_applicant_name,
        "audit_id":          audit_id,
        "status":            decision.lower(),
        "next_action":       notification_type,
        "notification_sent": notification_sent,
        "email_sent":        bool(recipient_email),
        "processing_time":   processing_time,
        "message":           "Decision processed successfully"
    }

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
