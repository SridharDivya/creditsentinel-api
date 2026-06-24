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

from typing import List, Optional

from feature_engine import compute_features
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import psycopg2
from psycopg2 import pool
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

TOTAL_APPLICATIONS = 15000

# =========================================================
# ENV VARS
# =========================================================
DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

MAIL_HOST           = os.getenv("MAIL_HOST")
MAIL_PORT           = int(os.getenv("MAIL_PORT", 2525))
MAIL_USERNAME       = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD       = os.getenv("MAIL_PASSWORD")
MAIL_FROM           = os.getenv("MAIL_FROM", "noreply@creditsentinel.com")
MAIL_TEST_RECIPIENT = os.getenv("MAIL_TEST_RECIPIENT", "test@inbox.mailtrap.io")

# =========================================================
# DB POOL
# =========================================================
db_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=30,
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    database=DB_CONFIG["database"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
)

def get_db_connection():
    return db_pool.getconn()

# =========================================================
# ASYNC AUDIT WORKER
# =========================================================
_audit_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audit")

def _audit_worker(payload: dict):
    conn = None
    try:
        conn   = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_trail
                (application_id, decision, decision_notes,
                 applicant_name, analyst_name, processing_time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING audit_id
        """, (
            payload["application_id"],
            payload["decision"],
            payload["notes"],
            payload["applicant_name"],
            payload.get("analyst_name", "Divya"),
            payload.get("processing_time", 0.0)
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_audit_executor, _audit_worker, payload)

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
        r = float(val)
        return default if (math.isnan(r) or math.isinf(r)) else r
    except:
        return default

def safe_int(val, default=0):
    try:
        r = float(val)
        return default if (math.isnan(r) or math.isinf(r)) else int(r)
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
# EMAIL
# =========================================================
def send_email(recipient: str, subject: str, body: str) -> bool:
    try:
        if not recipient or recipient.strip() == "":
            recipient = MAIL_TEST_RECIPIENT

        msg            = MIMEMultipart()
        msg["From"]    = MAIL_FROM
        msg["To"]      = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(MAIL_HOST, MAIL_PORT)
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_FROM, recipient, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False

# =========================================================
# SHARED HELPERS
# =========================================================
def get_risk_tier(risk_score: float) -> str:
    if risk_score < 0.4:    return "Low"
    elif risk_score < 0.65: return "Medium"
    else:                   return "High"

def get_status(risk_tier: str) -> str:
    return {"Low": "Approved", "Medium": "Under Review", "High": "Rejected"}.get(risk_tier, "Pending")

def get_foir(monthly_income: float, monthly_emi: float) -> float:
    return round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0.0

def get_emi_from_row(row) -> float:
    for col in ["existing_monthly_emi", "monthly_emi", "emi", "current_emi", "total_emi"]:
        val = safe_float(row.get(col, None), default=-1)
        if val >= 0:
            return val
    return 0.0

def get_decision_date(application_id: str) -> str:
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp) FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
        """, (application_id.strip().upper(),))
        row = cursor.fetchone()
        cursor.close()
        db_pool.putconn(conn)
        if row and row[0]:
            return row[0].isoformat()
        return ""
    except Exception:
        return ""

def get_real_status(application_id: str, risk_tier: str) -> str:
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT decision FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (application_id.strip().upper(),))
        row = cursor.fetchone()
        cursor.close()
        db_pool.putconn(conn)
        if row and row[0]:
            return {"APPROVE": "Approved", "REJECT": "Rejected", "REVIEW": "Under Review"}.get(
                str(row[0]).upper(), get_status(risk_tier)
            )
        return get_status(risk_tier)
    except Exception:
        return get_status(risk_tier)

def get_processing_time(application_id: str) -> float:
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT processing_time FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (application_id.strip().upper(),))
        row = cursor.fetchone()
        cursor.close()
        db_pool.putconn(conn)
        if row and row[0] is not None:
            return round(float(row[0]), 3)
        return 0.0
    except Exception:
        return 0.0

def generate_decision_note(decision: str, risk_score: float, risk_tier: str, cibil_score: int) -> str:
    if decision == "APPROVE":
        return f"Credit profile low risk — approved. Risk score: {risk_score} | CIBIL: {cibil_score} | Tier: {risk_tier}"
    elif decision == "REJECT":
        return f"Credit profile high risk — rejected. Risk score: {risk_score} | CIBIL: {cibil_score} | Tier: {risk_tier}"
    elif decision == "REVIEW":
        return f"Credit profile medium risk — sent for manual review. Risk score: {risk_score} | CIBIL: {cibil_score} | Tier: {risk_tier}"
    return f"Decision based on risk score: {risk_score} | CIBIL: {cibil_score}"

def compute_cibil_score(row) -> int:
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    foir               = safe_float(d.get("foir", 0))
    monthly_income     = safe_float(d.get("monthly_income", 0))
    loan_to_income     = safe_float(d.get("loan_to_income_ratio", 0))
    num_existing_loans = safe_float(d.get("num_existing_loans", 0))
    employment_years   = safe_float(d.get("employment_years", 0))

    score = 750.0
    if foir <= 30:   score += 40
    elif foir <= 40: score += 10
    elif foir <= 50: score -= 20
    elif foir <= 60: score -= 60
    else:            score -= 100

    if monthly_income >= 100000:   score += 50
    elif monthly_income >= 75000:  score += 35
    elif monthly_income >= 50000:  score += 20
    elif monthly_income >= 30000:  score += 5
    else:                          score -= 20

    if loan_to_income <= 2:   score += 30
    elif loan_to_income <= 4: score += 10
    elif loan_to_income <= 6: score -= 20
    else:                     score -= 50

    if num_existing_loans == 0:   score += 20
    elif num_existing_loans == 1: score += 5
    elif num_existing_loans == 2: score -= 15
    else:                         score -= 30 * (num_existing_loans - 2)

    if employment_years >= 10:   score += 40
    elif employment_years >= 5:  score += 25
    elif employment_years >= 3:  score += 10
    elif employment_years >= 1:  score -= 5
    else:                        score -= 25

    return max(300, min(900, int(score)))

def get_ml_credit_score(risk_score: float) -> int:
    return int(300 + (1 - risk_score) * 600)

# =========================================================
# FEATURE CACHE
# =========================================================
_feature_cache: dict = {}
_feature_cache_lock  = threading.Lock()
FEATURE_CACHE_MAX    = 500

def _get_cached_features(application_id: str) -> dict:
    with _feature_cache_lock:
        if application_id in _feature_cache:
            return _feature_cache[application_id]
    features = compute_features(application_id)
    with _feature_cache_lock:
        if len(_feature_cache) >= FEATURE_CACHE_MAX:
            del _feature_cache[next(iter(_feature_cache))]
        _feature_cache[application_id] = features
    return features

def generate_risk_score(application_id: str) -> dict:
    try:
        features_dict     = _get_cached_features(application_id)
        filtered_features = {f: features_dict.get(f, 0) for f in MODEL_FEATURES}
        features_df       = pd.DataFrame([filtered_features])[MODEL_FEATURES]
        features_df       = features_df.fillna(0).replace([np.inf, -np.inf], 0).astype(float)
        risk_score        = round(float(model.predict_proba(features_df)[:, 1][0]), 4)
        return {"risk_score": risk_score, "risk_tier": get_risk_tier(risk_score)}
    except Exception:
        return {"risk_score": 0.0, "risk_tier": "Low"}

# =========================================================
# REQUEST MODELS
# =========================================================
class ScoreRequest(BaseModel):
    application_id: str

class BatchScoreRequest(BaseModel):
    application_ids: List[str]

class DecisionRequest(BaseModel):
    decision:     str
    notes:        Optional[str] = ""
    analyst_name: str
    timestamp:    Optional[str] = None

# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/api/score")
def score_application(req: ScoreRequest):
    start_time = time.time()
    try:
        result = generate_risk_score(req.application_id)
        processing_time = round(time.time() - start_time, 4)
        return {
            "application_id": req.application_id,
            "risk_score": result["risk_score"],
            "risk_tier": result["risk_tier"],
            "processing_time_seconds": processing_time,
            "latency_ms": round(processing_time * 1000, 2),
        }
    except Exception as e:
        return {"application_id": req.application_id, "error": str(e)}

@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(req.application_ids))) as ex:
        futures = {ex.submit(generate_risk_score, app_id): app_id for app_id in req.application_ids}
        results = []
        for future, app_id in futures.items():
            r = future.result()
            results.append({"application_id": app_id, "risk_score": r["risk_score"], "risk_tier": r["risk_tier"]})
    
    processing_time = round(time.time() - start_time, 4)
    return {
        "total_applications": len(results), 
        "results": results, 
        "processing_time_seconds": processing_time, 
        "latency_ms": round(processing_time * 1000, 2)
    }

@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    try:
        rows_list = []
        for i in range(offset, offset + limit):
            row = applications_df.iloc[i % len(applications_df)].copy()
            row["application_id"] = f"APP-{i+1:06d}"
            rows_list.append(row)
        subset  = pd.DataFrame(rows_list)
        app_ids = [safe_str(row.get("application_id", "")) for _, row in subset.iterrows()]

        with ThreadPoolExecutor(max_workers=min(8, len(app_ids))) as ex:
            score_map = {app_id: future.result() for app_id, future in ((app_id, ex.submit(generate_risk_score, app_id)) for app_id in app_ids)}

        applications = []
        for _, row in subset.iterrows():
            app_id         = safe_str(row.get("application_id", ""))
            result         = score_map[app_id]
            risk_score     = result["risk_score"]
            risk_tier      = result["risk_tier"]
            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = get_emi_from_row(row)
            p_time         = get_processing_time(app_id)

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
                "application_status": get_real_status(app_id, risk_tier),
                "created_at":         safe_str(row.get("created_at", row.get("application_date", ""))),
                "decision_date":      get_decision_date(app_id),
                "processing_time_seconds": p_time,
                "latency_ms":         round(p_time * 1000, 2)
            })
        return {"total": TOTAL_APPLICATIONS, "applications": applications}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):
    try:
        matched = applications_df[applications_df["application_id"].astype(str) == str(application_id)]
        if len(matched) == 0:
            numeric = int(str(application_id).split("-")[-1]) - 1
            row     = applications_df.iloc[numeric % len(applications_df)].copy()
            row["application_id"] = application_id
        else:
            row = matched.iloc[0]

        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = get_emi_from_row(row)
        foir           = round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0

        score_data   = generate_risk_score(application_id)
        risk_score   = score_data["risk_score"]
        risk_tier    = score_data["risk_tier"]
        cibil_score  = compute_cibil_score(row)
        credit_score = get_ml_credit_score(risk_score)
        p_time       = get_processing_time(application_id)

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
            "application_status": get_real_status(application_id, risk_tier),
            "date_applied":       safe_str(row.get("application_date", row.get("date_applied", ""))),
            "decision_date":      get_decision_date(application_id),
            "processing_time_seconds": p_time,
            "latency_ms":         round(p_time * 1000, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):
    try:
        clean_id = str(application_id).strip().upper()
        matched = applications_df[applications_df["application_id"].astype(str).str.strip().str.upper() == clean_id]
        csv_applicant_name = safe_str(matched.iloc[0]["applicant_name"]) if len(matched) > 0 else "Unknown Applicant"

        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT audit_id, decision, decision_notes, timestamp, applicant_name, analyst_name, processing_time
            FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC
        """, (clean_id,))
        rows = cursor.fetchall()
        cursor.close()
        db_pool.putconn(conn)

        if not rows:
            start_time = time.time()
            score_data  = generate_risk_score(application_id)
            decision    = {"Low": "APPROVE", "Medium": "REVIEW", "High": "REJECT"}.get(score_data["risk_tier"], "REVIEW")
            processing_time = round(time.time() - start_time, 4)

            payload = {
                "application_id":  application_id,
                "decision":        decision,
                "notes":           "Auto-generated",
                "applicant_name":  csv_applicant_name,
                "analyst_name":    "Divya",
                "processing_time": processing_time
            }
            real_audit_id = _audit_worker(payload)
            return {"history": [{
                "audit_id":       real_audit_id,
                "decision":       decision,
                "notes":          "Auto-generated",
                "timestamp":      datetime.now().isoformat(),
                "applicant_name": csv_applicant_name,
                "analyst_name":   "Divya",
                "processing_time_seconds": processing_time,
                "latency_ms":     round(processing_time * 1000, 2)
            }]}

        history = []
        for row in rows:
            p_time = float(row[6]) if row[6] else 0.0
            history.append({
                "audit_id":       row[0],
                "decision":       row[1],
                "notes":          row[2],
                "timestamp":      row[3].isoformat() if row[3] else None,
                "applicant_name": safe_str(row[4]) if row[4] else csv_applicant_name,
                "analyst_name":   safe_str(row[5]) if row[5] else "Divya",
                "processing_time_seconds": p_time,
                "latency_ms":     round(p_time * 1000, 2)
            })
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/applications/{application_id}/process-decision")
async def process_decision(application_id: str, req: DecisionRequest):
    decision_start = time.time()
    decision_map = {"APPROVE": "APPROVE", "APPROVED": "APPROVE", "REJECT": "REJECT", "REJECTED": "REJECT", "REVIEW": "REVIEW"}
    decision = str(req.decision).strip().upper()
    if decision not in decision_map:
        return {"status": "failed", "error": "Invalid decision"}

    decision     = decision_map[decision]
    search_id    = str(application_id).strip().upper()
    matched = applications_df[applications_df["application_id"].astype(str).str.strip().str.upper() == search_id]
    real_applicant_name = safe_str(matched.iloc[0].get("applicant_name", "Unknown Applicant")) if len(matched) > 0 else "Unknown Applicant"

    processing_time = round(time.time() - decision_start, 4)
    latency_ms = round(processing_time * 1000, 2)

    audit_payload = {
        "application_id":  application_id,
        "decision":        decision,
        "notes":           req.notes or "Processed",
        "applicant_name":  real_applicant_name,
        "analyst_name":    req.analyst_name or "Divya",
        "processing_time": processing_time
    }
    audit_id = await fire_and_forget_audit(audit_payload)

    return {
        "application_id": application_id,
        "status": decision.lower(),
        "processing_time_seconds": processing_time,
        "latency_ms": latency_ms,
        "audit_id": audit_id
    }

@app.get("/api/portfolio/summary")
def portfolio_summary():
    start = time.time()
    try:
        elapsed = round(time.time() - start, 4)
        return {
            "total_applications": TOTAL_APPLICATIONS,
            "processing_time_seconds": elapsed,
            "latency_ms": round(elapsed * 1000, 2)
        }
    except Exception as e:
        return {"error": str(e)}
