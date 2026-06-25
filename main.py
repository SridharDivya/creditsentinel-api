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
print("CSV COLUMNS:", list(applications_df.columns))

# Pre-build O(1) lookup index — avoids full DataFrame scan on every request
_csv_index: dict = {
    str(row.get("application_id", "")).strip().upper(): row
    for _, row in applications_df.iterrows()
}
print(f"✅ CSV index built: {len(_csv_index)} entries")

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
print("✅ Connection Pool Initialized")

def get_db_connection():
    return db_pool.getconn()

try:
    _t = get_db_connection()
    db_pool.putconn(_t)
    print("✅ PostgreSQL Connected")
except Exception as e:
    print(f"❌ PostgreSQL Connection Failed: {e}")

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
                 applicant_name, analyst_name, timestamp,
                 latency_ms, email_sent)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
            RETURNING audit_id
        """, (
            payload["application_id"],
            payload["decision"],
            payload["notes"],
            payload["applicant_name"],
            payload.get("analyst_name", ""),
            payload.get("latency_ms", 0),
            payload.get("email_sent", False),
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
            print(f"[EMAIL] No applicant email — using test recipient: {recipient}")

        print(f"[EMAIL] Sending to {recipient} | host={MAIL_HOST} port={MAIL_PORT}")

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

        print(f"✅ Email sent successfully to {recipient}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False

# =========================================================
# SHARED HELPERS
# =========================================================
def get_risk_tier(risk_score: float) -> str:
    if risk_score < 0.4:   return "Low"
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

_decision_date_cache: dict = {}
_decision_date_lock        = threading.Lock()
DECISION_DATE_TTL          = 60   # seconds

def get_decision_date(application_id: str) -> str:
    now = time.time()
    with _decision_date_lock:
        entry = _decision_date_cache.get(application_id)
        if entry and (now - entry["ts"]) < DECISION_DATE_TTL:
            return entry["val"]
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(timestamp) FROM audit_trail WHERE application_id = %s",
            (application_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        db_pool.putconn(conn)
        val = row[0].isoformat() if row and row[0] else ""
    except Exception:
        val = ""
    with _decision_date_lock:
        _decision_date_cache[application_id] = {"val": val, "ts": now}
    return val

_real_status_cache: dict = {}
_real_status_lock          = threading.Lock()
REAL_STATUS_TTL            = 60   # seconds

def get_real_status(application_id: str, risk_tier: str) -> str:
    now = time.time()
    with _real_status_lock:
        entry = _real_status_cache.get(application_id)
        if entry and (now - entry["ts"]) < REAL_STATUS_TTL:
            return entry["val"]
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT decision FROM audit_trail WHERE application_id = %s ORDER BY timestamp DESC LIMIT 1",
            (application_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        db_pool.putconn(conn)
        val = {"APPROVE": "Approved", "REJECT": "Rejected", "REVIEW": "Under Review"}.get(
            str(row[0]).upper(), get_status(risk_tier)
        ) if row and row[0] else get_status(risk_tier)
    except Exception:
        val = get_status(risk_tier)
    with _real_status_lock:
        _real_status_cache[application_id] = {"val": val, "ts": now}
    return val

# =========================================================
# CIBIL SCORE
# =========================================================
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
_feature_cache: dict      = {}
_feature_cache_lock       = threading.Lock()
FEATURE_CACHE_MAX         = 500

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

# =========================================================
# CORE: ML MODEL
# =========================================================
def generate_risk_score(application_id: str) -> dict:
    try:
        features_dict     = _get_cached_features(application_id)
        filtered_features = {f: features_dict.get(f, 0) for f in MODEL_FEATURES}
        features_df       = pd.DataFrame([filtered_features])[MODEL_FEATURES]
        features_df       = features_df.fillna(0).replace([np.inf, -np.inf], 0).astype(float)
        risk_score        = round(float(model.predict_proba(features_df)[:, 1][0]), 4)
        return {"risk_score": risk_score, "risk_tier": get_risk_tier(risk_score)}
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
    decision:     str
    notes:        Optional[str] = ""
    analyst_name: str                   # required — who made the decision

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status":             "ok",
        "model_loaded":       True,
        "total_applications": TOTAL_APPLICATIONS,
        "cibil_source":       "computed_from_foir_income_lti_loans_employment",
    }

# =========================================================
# SCORE SINGLE
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):
    start_time = time.time()
    try:
        result     = generate_risk_score(req.application_id)
        latency_ms = (time.time() - start_time) * 1000
        log_entry  = {
            "timestamp": datetime.now().isoformat(), "application_id": req.application_id,
            "risk_score": result["risk_score"], "risk_tier": result["risk_tier"],
            "latency_ms": round(latency_ms, 2), "status": "success",
        }
        with open("model_predictions.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return {
            "application_id": req.application_id, "model_loaded": True,
            "risk_score": result["risk_score"], "risk_tier": result["risk_tier"],
            "features_used": len(MODEL_FEATURES), "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        with open("model_predictions.log", "a") as f:
            f.write(json.dumps({"timestamp": datetime.now().isoformat(),
                "application_id": req.application_id, "latency_ms": round(latency_ms, 2),
                "status": "error", "error": str(e)}) + "\n")
        return {"application_id": req.application_id, "model_loaded": False, "error": str(e)}

# =========================================================
# SCORE BATCH
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    with ThreadPoolExecutor(max_workers=min(8, len(req.application_ids))) as ex:
        futures = {ex.submit(generate_risk_score, app_id): app_id for app_id in req.application_ids}
        results = []
        for future, app_id in futures.items():
            r = future.result()
            results.append({"application_id": app_id, "risk_score": r["risk_score"], "risk_tier": r["risk_tier"]})
    return {"total_applications": len(results), "results": results}

# =========================================================
# APPLICATIONS LIST
# =========================================================
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
            score_map = {
                app_id: future.result()
                for app_id, future in
                ((app_id, ex.submit(generate_risk_score, app_id)) for app_id in app_ids)
            }

        # ── Batch fetch status + decision_date in 2 queries instead of N×2 ──
        try:
            conn   = get_db_connection()
            cursor = conn.cursor()
            placeholders = ",".join(["%s"] * len(app_ids))
            cursor.execute(
                f"SELECT application_id, decision, MAX(timestamp) "
                f"FROM audit_trail WHERE application_id IN ({placeholders}) "
                f"GROUP BY application_id, decision ORDER BY MAX(timestamp) DESC",
                app_ids
            )
            db_rows = cursor.fetchall()
            cursor.close()
            db_pool.putconn(conn)

            # Build lookup maps
            _status_map  = {}
            _date_map    = {}
            for db_app_id, db_decision, db_ts in db_rows:
                if db_app_id not in _status_map:
                    _status_map[db_app_id] = {"APPROVE": "Approved", "REJECT": "Rejected",
                        "REVIEW": "Under Review"}.get(str(db_decision).upper(), "Pending")
                    _date_map[db_app_id]   = db_ts.isoformat() if db_ts else ""
        except Exception:
            _status_map = {}
            _date_map   = {}

        applications = []
        for _, row in subset.iterrows():
            app_id         = safe_str(row.get("application_id", ""))
            result         = score_map[app_id]
            risk_score     = result["risk_score"]
            risk_tier      = result["risk_tier"]
            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = get_emi_from_row(row)

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
                "application_status": _status_map.get(app_id, get_status(risk_tier)),
                "created_at":         safe_str(row.get("created_at", row.get("application_date", ""))),
                "decision_date":      _date_map.get(app_id, ""),
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
        matched = applications_df[applications_df["application_id"].astype(str) == str(application_id)]
        if len(matched) == 0:
            try:
                numeric = int(str(application_id).split("-")[-1]) - 1
                row     = applications_df.iloc[numeric % len(applications_df)].copy()
                row["application_id"] = application_id
            except Exception:
                return {"error": "Application not found"}
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
            "application_status": get_real_status(application_id, risk_tier),
            "date_applied":       safe_str(row.get("application_date", row.get("date_applied", ""))),
            "decision_date":      get_decision_date(application_id),
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# SCORE + CIBIL CACHE — avoid recomputing on every history call
# TTL: 1 hour per application_id
# =========================================================
_score_cache: dict      = {}
_cibil_cache: dict      = {}
_score_cache_lock       = threading.Lock()
SCORE_CACHE_TTL         = 3600   # seconds

def _get_cached_risk_score(application_id: str) -> dict:
    now = time.time()
    with _score_cache_lock:
        entry = _score_cache.get(application_id)
        if entry and (now - entry["ts"]) < SCORE_CACHE_TTL:
            return entry["data"]
    result = generate_risk_score(application_id)
    with _score_cache_lock:
        _score_cache[application_id] = {"data": result, "ts": now}
    return result

def _get_cached_cibil(application_id: str, csv_row) -> int:
    with _score_cache_lock:
        if application_id in _cibil_cache:
            return _cibil_cache[application_id]
    score = compute_cibil_score(csv_row) if csv_row is not None else 650
    with _score_cache_lock:
        _cibil_cache[application_id] = score
    return score

# =========================================================
# CREDIT-SCORE BASED NOTE GENERATOR
# =========================================================
def get_credit_based_note(decision: str, cibil_score: int, risk_score: float, risk_tier: str) -> str:
    if decision in ("APPROVE", "APPROVED"):
        if cibil_score >= 750:
            return (f"Application approved. Excellent credit score of {cibil_score} with low risk profile "
                    f"(score: {risk_score}). Applicant meets all creditworthiness criteria.")
        else:
            return (f"Application approved. Credit score {cibil_score} is satisfactory. "
                    f"Risk tier: {risk_tier} (score: {risk_score}). Standard terms applied.")
    elif decision in ("REJECT", "REJECTED"):
        if cibil_score < 600:
            return (f"Application rejected. Credit score {cibil_score} is below minimum threshold of 600. "
                    f"High risk profile (score: {risk_score}). Applicant advised to improve credit standing.")
        else:
            return (f"Application rejected. Despite credit score of {cibil_score}, risk assessment indicates "
                    f"{risk_tier.lower()} risk (score: {risk_score}). Additional risk factors identified.")
    else:  # REVIEW
        return (f"Application flagged for manual review. Credit score {cibil_score} requires further assessment. "
                f"Risk tier: {risk_tier} (score: {risk_score}). Assigned to senior analyst for evaluation.")

# =========================================================
# HISTORY — GET /api/applications/{id}/history
#
# - analyst_name: read from audit_trail DB (whoever submitted
#   the decision via process-decision). Falls back to "SYSTEM"
#   only when no analyst was recorded.
# - notes: auto-generated based on credit score per decision
# - email_to: returned in response
# - email report fires silently on every call
# - latency_ms: returned at top level
# =========================================================
# ── In-memory history cache: avoids repeat DB hits ──────────
_history_cache: dict = {}
_history_cache_lock  = threading.Lock()
HISTORY_CACHE_TTL    = 30   # seconds — short TTL so new decisions show up quickly

@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):
    history_start = time.time()

    try:
        # Normalise once
        clean_id = application_id.strip().upper()

        # ── 1. CSV row lookup — O(1) via pre-built index ─────────
        csv_row = _csv_index.get(clean_id)
        if csv_row is None:
            try:
                numeric = int(clean_id.split("-")[-1]) - 1
                csv_row = applications_df.iloc[numeric % len(applications_df)].copy()
                csv_row["application_id"] = application_id
            except Exception:
                csv_row = None

        csv_applicant_name   = safe_str(csv_row["applicant_name"]) if csv_row is not None else "Unknown Applicant"
        csv_application_date = safe_str(csv_row.get("application_date", "")) if csv_row is not None else ""
        csv_created_at       = safe_str(csv_row.get("created_at",   csv_application_date)) if csv_row is not None else ""
        csv_submitted_at     = safe_str(csv_row.get("submitted_at", csv_application_date)) if csv_row is not None else ""

        # ── 2. Email resolve — fast, no DB ───────────────────────
        email_to = MAIL_TEST_RECIPIENT
        if csv_row is not None:
            for col in ["email", "email_address", "applicant_email", "mail"]:
                v = safe_str(csv_row.get(col, ""))
                if v.strip():
                    email_to = v.strip()
                    break

        # ── 3. Check history cache first ─────────────────────────
        now = time.time()
        with _history_cache_lock:
            cached = _history_cache.get(clean_id)
            if cached and (now - cached["ts"]) < HISTORY_CACHE_TTL:
                result = dict(cached["data"])
                result["latency_ms"] = round((time.time() - history_start) * 1000, 2)
                return result

        # ── 4. Score + CIBIL — from cache, not recomputed ────────
        score_data  = _get_cached_risk_score(application_id)
        risk_score  = score_data["risk_score"]
        risk_tier   = score_data["risk_tier"]
        cibil_score = _get_cached_cibil(application_id, csv_row)

        # ── 5. Single DB query — no UPPER/TRIM overhead ──────────
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT audit_id, decision, decision_notes, timestamp, applicant_name, analyst_name "
            "FROM audit_trail WHERE application_id = %s ORDER BY timestamp DESC LIMIT 50",
            (application_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        db_pool.putconn(conn)

        # ── Auto-insert if no history exists ─────────────────────
        if not rows:
            try:
                decision    = {"Low": "APPROVE", "Medium": "REVIEW", "High": "REJECT"}.get(risk_tier, "REVIEW")
                status_note = get_credit_based_note(decision, cibil_score, risk_score, risk_tier)

                payload = {
                    "application_id": application_id,
                    "decision":       decision,
                    "notes":          status_note,
                    "applicant_name": csv_applicant_name,
                    "analyst_name":   "SYSTEM",   # no real analyst yet — first auto-entry
                }
                real_audit_id = _audit_worker(payload)
                latency_ms    = round((time.time() - history_start) * 1000, 2)

                # fire email in background — non-blocking
                _auto_body = (
                    f"Decision History for {application_id}\n"
                    f"Applicant : {csv_applicant_name}\n"
                    f"Decision  : {decision}\n"
                    f"Analyst   : SYSTEM\n"
                    f"Date      : {csv_application_date or 'N/A'}\n"
                    f"Notes     : {status_note}\n"
                )
                threading.Thread(
                    target=send_email,
                    args=(email_to, f"History Report – {application_id}", _auto_body),
                    daemon=True,
                ).start()

                return {
                    "history": [{
                        "audit_id":         real_audit_id,
                        "decision":         decision,
                        "notes":            status_note,
                        "timestamp":        csv_application_date or datetime.now().isoformat(),
                        "applicant_name":   csv_applicant_name,
                        "analyst_name":     "SYSTEM",
                        "application_date": csv_application_date,
                        "decision_date":    csv_application_date,
                        "created_at":       csv_created_at,
                        "submitted_at":     csv_submitted_at,
                    }],
                    "email_report": True,
                    "email_to":     email_to,
                    "latency_ms":   latency_ms,
                }
            except Exception as insert_err:
                print(f"[AUDIT AUTO-INSERT ERROR] {insert_err}")
                return {"history": [], "email_report": True, "email_to": email_to, "latency_ms": 0}

        # ── Build history list ───────────────────────────────────
        history    = []
        now_ts     = time.time()

        for i, row in enumerate(rows):
            # analyst_name: read directly from DB
            raw_analyst  = row[5] if len(row) > 5 else None
            analyst_str  = str(raw_analyst).strip() if raw_analyst is not None else ""
            analyst_name = analyst_str if analyst_str and analyst_str.lower() not in ["none", "null", ""] else "SYSTEM"

            db_appl_val  = row[4]
            db_appl_str  = str(db_appl_val).strip() if db_appl_val is not None else ""
            final_applicant_name = (
                csv_applicant_name
                if (db_appl_val is None or db_appl_str == "" or db_appl_str.lower() in ["none", "null"])
                else safe_str(db_appl_val)
            )

            ts_obj      = row[3]
            ts_iso      = ts_obj.isoformat() if ts_obj else None
            db_decision = safe_str(row[1])
            smart_note  = get_credit_based_note(db_decision, cibil_score, risk_score, risk_tier)

            # per-record latency_ms for latency trend chart:
            # most recent record (i=0) -> elapsed request time
            # older records -> gap between consecutive decisions in ms
            if i == 0:
                rec_latency_ms = round((now_ts - history_start) * 1000, 2)
            elif ts_obj and rows[i - 1][3]:
                rec_latency_ms = round(abs((rows[i - 1][3] - ts_obj).total_seconds() * 1000), 2)
            else:
                rec_latency_ms = 0.0

            history.append({
                "audit_id":         row[0],
                "decision":         db_decision,
                "notes":            smart_note,
                "timestamp":        ts_iso,
                "applicant_name":   final_applicant_name,
                "analyst_name":     analyst_name,
                "application_date": csv_application_date,
                "decision_date":    ts_iso or csv_application_date,
                "created_at":       csv_created_at,
                "submitted_at":     csv_submitted_at,
                "latency_ms":       rec_latency_ms,
            })

        latency_ms = round((time.time() - history_start) * 1000, 2)

        # ── Email fires in background — non-blocking ────────────
        def _send_history_email():
            lines = [
                f"Decision History Report – {application_id}",
                f"Applicant : {csv_applicant_name}",
                f"Total Records: {len(history)}",
                "",
            ]
            for rec in history:
                lines.append(
                    f"  [{rec['timestamp'] or 'N/A'}]  {rec['decision']}  "
                    f"by {rec['analyst_name']}  |  {rec['notes'] or ''}"
                )
            send_email(email_to, f"History Report – {application_id}", "\n".join(lines))
        threading.Thread(target=_send_history_email, daemon=True).start()

        response = {
            "history":      history,
            "email_report": True,
            "email_to":     email_to,
            "latency_ms":   latency_ms,
        }

        # ── Store in history cache for next 30s ──────────────────
        with _history_cache_lock:
            _history_cache[clean_id] = {"data": response, "ts": time.time()}

        return response

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# PROCESS DECISION — POST /api/applications/{id}/process-decision
# =========================================================
@app.post("/api/applications/{application_id}/process-decision")
async def process_decision(application_id: str, req: DecisionRequest):

    decision_start = time.time()

    decision_map = {
        "APPROVE": "APPROVE", "APPROVED": "APPROVE",
        "REJECT":  "REJECT",  "REJECTED": "REJECT",
        "REVIEW":  "REVIEW",
    }

    decision = str(req.decision).strip().upper()
    if decision not in decision_map:
        return {"status": "failed", "error": "Invalid decision. Allowed: APPROVE, REJECT, REVIEW"}

    decision     = decision_map[decision]
    notes        = req.notes or ""
    analyst_name = req.analyst_name or ""
    conn         = None
    search_id    = str(application_id).strip().upper()

    matched = applications_df[
        applications_df["application_id"].astype(str).str.strip().str.upper() == search_id
    ]
    if len(matched) == 0:
        try:
            numeric = int(search_id.split("-")[-1]) - 1
            matched = applications_df.iloc[[numeric % len(applications_df)]].copy()
            matched["application_id"] = application_id
        except Exception:
            return JSONResponse(status_code=404, content={
                "status": "failed", "error": f"Application ID {application_id} not found"
            })

    real_applicant_name = safe_str(matched.iloc[0].get("applicant_name", "Unknown Applicant"))

    recipient_email = ""
    for col in ["email", "email_address", "applicant_email", "mail", "contact_email"]:
        val = safe_str(matched.iloc[0].get(col, ""))
        if val.strip():
            recipient_email = val.strip()
            break

    if not recipient_email:
        recipient_email = MAIL_TEST_RECIPIENT
        print(f"[EMAIL] No CSV email for {application_id} — using: {recipient_email}")

    notification_sent = False
    notification_type = None

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        if decision == "APPROVE":
            cursor.execute("""
                UPDATE applications SET application_status = 'approved', updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "approval_email"

        elif decision == "REJECT":
            cursor.execute("""
                UPDATE applications SET application_status = 'rejected', updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "rejection_email"

        elif decision == "REVIEW":
            cursor.execute("""
                UPDATE applications
                SET application_status = 'under_review', assigned_reviewer = 'TEAM_LEAD',
                    updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(TRIM(application_id)) = %s
            """, (search_id,))
            notification_sent = True
            notification_type = "internal_review_notification"

        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        conn = None

        # Email fires in background — non-blocking so response returns fast
        email_sent = True   # optimistic — thread handles actual sending
        if decision == "APPROVE":
            _subj = "Loan Application Approved"
            _body = f"Hello {real_applicant_name},\n\nCongratulations! Your loan application {application_id} has been APPROVED.\n\nRegards,\nCreditSentinel Team"
        elif decision == "REJECT":
            _subj = "Loan Application Rejected"
            _body = f"Hello {real_applicant_name},\n\nYour loan application {application_id} has been REJECTED.\n\nReason: {notes}\n\nRegards,\nCreditSentinel Team"
        elif decision == "REVIEW":
            _subj = "Application Under Review"
            _body = f"Hello {real_applicant_name},\n\nYour loan application {application_id} is currently UNDER REVIEW.\nOur team will contact you shortly.\n\nRegards,\nCreditSentinel Team"
        else:
            _subj = _body = None
            email_sent = False
        if _subj:
            threading.Thread(target=send_email, args=(recipient_email, _subj, _body), daemon=True).start()

    except Exception as e:
        if conn:
            conn.rollback()
            try: cursor.close()
            except: pass
            db_pool.putconn(conn)
        return JSONResponse(status_code=500, content={
            "status": "failed", "application_id": application_id, "error": str(e)
        })

    audit_payload = {
        "application_id": application_id,
        "decision":       decision,
        "notes":          notes,
        "applicant_name": real_applicant_name,
        "analyst_name":   analyst_name,
        "latency_ms":     round((time.time() - decision_start) * 1000, 2),
        "email_sent":     True,
    }
    audit_id   = await asyncio.create_task(fire_and_forget_audit(audit_payload))
    latency_ms = round((time.time() - decision_start) * 1000, 2)

    # Invalidate all caches for this application so next read is fresh
    _cid = application_id.strip().upper()
    with _history_cache_lock:
        _history_cache.pop(_cid, None)
    with _decision_date_lock:
        _decision_date_cache.pop(application_id, None)
    with _real_status_lock:
        _real_status_cache.pop(application_id, None)

    return {
        "application_id":    application_id,
        "applicant_name":    real_applicant_name,
        "analyst_name":      analyst_name,
        "audit_id":          audit_id,
        "status":            decision.lower(),
        "next_action":       notification_type,
        "notification_sent": notification_sent,
        "email_sent":        email_sent,
        "email_to":          recipient_email,
        "latency_ms":        latency_ms,
        "message":           "Decision processed successfully",
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
        score += np.select([foir<=30, foir<=40, foir<=50, foir<=60], [40,10,-20,-60], default=-100)
        score += np.select([monthly_income>=100000, monthly_income>=75000,
                            monthly_income>=50000,  monthly_income>=30000], [50,35,20,5], default=-20)
        score += np.select([loan_to_income<=2, loan_to_income<=4, loan_to_income<=6], [30,10,-20], default=-50)
        extra_penalty = np.where(num_existing_loans>2, -30*(num_existing_loans-2), 0)
        score += np.select([num_existing_loans==0, num_existing_loans==1, num_existing_loans==2],
                           [20,5,-15], default=extra_penalty)
        score += np.select([employment_years>=10, employment_years>=5,
                            employment_years>=3,  employment_years>=1], [40,25,10,-5], default=-25)

        score  = score.clip(300, 900).astype(int)
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
            "execution_time_seconds": elapsed,
        }

    except Exception as e:
        err = traceback.format_exc()
        print("PORTFOLIO ERROR:", err)
        return {"error": str(e), "detail": err}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
