from fastapi import FastAPI, BackgroundTasks
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

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

TOTAL_APPLICATIONS = 15000

# =========================================================
# MODEL FEATURES
# =========================================================
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = list(model.feature_name_())

# =========================================================
# POSTGRESQL CONNECTION POOL
# =========================================================
DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

db_pool = pool.ThreadedConnectionPool(
    minconn=5, maxconn=30, **DB_CONFIG
)
print("✅ Connection Pool Initialized")

def get_db_connection():
    return db_pool.getconn()

try:
    _t = get_db_connection(); db_pool.putconn(_t)
    print("✅ PostgreSQL Connected")
except Exception as e:
    print(f"❌ PostgreSQL Connection Failed: {e}")

# =========================================================
# BACKGROUND THREAD POOLS
# Score / audit / log each get isolated workers so they
# never steal slots from each other under load.
# =========================================================
_score_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="score")
_audit_executor = ThreadPoolExecutor(max_workers=4,  thread_name_prefix="audit")
_log_executor   = ThreadPoolExecutor(max_workers=2,  thread_name_prefix="log")

# =========================================================
# SAFE HELPERS
# =========================================================
def safe_float(val, default=0.0):
    try:
        r = float(val)
        return default if (math.isnan(r) or math.isinf(r)) else r
    except: return default

def safe_int(val, default=0):
    try:
        r = float(val)
        return default if (math.isnan(r) or math.isinf(r)) else int(r)
    except: return default

def safe_str(val, default=""):
    try:
        if val is None: return default
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return default
        return str(val)
    except: return default

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

def get_ml_credit_score(risk_score: float) -> int:
    return int(300 + (1 - risk_score) * 600)

# =========================================================
# CIBIL SCORE — pure function, no pandas overhead
# =========================================================
def compute_cibil_score(row) -> int:
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    foir   = safe_float(d.get("foir", 0))
    income = safe_float(d.get("monthly_income", 0))
    lti    = safe_float(d.get("loan_to_income_ratio", 0))
    loans  = safe_float(d.get("num_existing_loans", 0))
    emp    = safe_float(d.get("employment_years", 0))
    s = 750.0
    s += 40 if foir<=30 else 10 if foir<=40 else -20 if foir<=50 else -60 if foir<=60 else -100
    s += 50 if income>=100000 else 35 if income>=75000 else 20 if income>=50000 else 5 if income>=30000 else -20
    s += 30 if lti<=2 else 10 if lti<=4 else -20 if lti<=6 else -50
    s += 20 if loans==0 else 5 if loans==1 else -15 if loans==2 else -30*(loans-2)
    s += 40 if emp>=10 else 25 if emp>=5 else 10 if emp>=3 else -5 if emp>=1 else -25
    return max(300, min(900, int(s)))

# =========================================================
# LRU CACHE HELPERS (thread-safe bounded OrderedDict)
# =========================================================
_CACHE_MAX = 20000   # large enough to hold all warmed IDs

_feature_cache      = OrderedDict()
_feature_cache_lock = threading.Lock()
_risk_cache         = OrderedDict()
_risk_cache_lock    = threading.Lock()

def _lru_get(cache, lock, key):
    with lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    return None

def _lru_set(cache, lock, key, value):
    with lock:
        if key in cache:
            cache.move_to_end(key)
        else:
            if len(cache) >= _CACHE_MAX:
                cache.popitem(last=False)
            cache[key] = value

def _get_cached_features(application_id: str) -> dict:
    hit = _lru_get(_feature_cache, _feature_cache_lock, application_id)
    if hit is not None:
        return hit
    features = compute_features(application_id)
    _lru_set(_feature_cache, _feature_cache_lock, application_id, features)
    return features

# =========================================================
# CORE: ML INFERENCE  (risk-score cache → feature cache)
# =========================================================
def generate_risk_score(application_id: str) -> dict:
    hit = _lru_get(_risk_cache, _risk_cache_lock, application_id)
    if hit is not None:
        return hit
    try:
        features_dict = _get_cached_features(application_id)
        arr = np.array([[features_dict.get(f, 0.0) for f in MODEL_FEATURES]], dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        features_df = pd.DataFrame(arr, columns=MODEL_FEATURES)
        risk_score  = round(float(model.predict_proba(features_df)[:, 1][0]), 4)
        risk_tier   = get_risk_tier(risk_score)
        result      = {"risk_score": risk_score, "risk_tier": risk_tier}
    except Exception:
        print(traceback.format_exc())
        result = {"risk_score": 0.0, "risk_tier": "Low"}
    _lru_set(_risk_cache, _risk_cache_lock, application_id, result)
    return result

# =========================================================
# APPLICATION LOOKUP INDEX  (O(1) dict, built at startup)
# =========================================================
_app_index: dict = {}        # normalised_id → row Series
_cibil_cache: dict = {}      # normalised_id → int
_app_meta: dict = {}         # APP-XXXXXX   → pre-built response dict

def _build_app_index():
    global _app_index, _cibil_cache
    idx, cibil = {}, {}
    for _, row in applications_df.iterrows():
        raw_id = safe_str(row.get("application_id", ""))
        if raw_id:
            key = raw_id.strip().upper()
            idx[key]   = row
            cibil[key] = compute_cibil_score(row)
    _app_index   = idx
    _cibil_cache = cibil
    print(f"✅ App index built: {len(_app_index)} entries")

_build_app_index()

def _lookup_row(application_id: str):
    key = str(application_id).strip().upper()
    if key in _app_index:
        return _app_index[key], application_id
    try:
        numeric = int(key.split("-")[-1]) - 1
        row = applications_df.iloc[numeric % len(applications_df)].copy()
        row["application_id"] = application_id
        return row, application_id
    except Exception:
        return None, None

def _lookup_cibil(application_id: str) -> int:
    key = str(application_id).strip().upper()
    if key in _cibil_cache:
        return _cibil_cache[key]
    row, _ = _lookup_row(application_id)
    return compute_cibil_score(row) if row is not None else 300

# =========================================================
# STARTUP CACHE WARM-UP
# Pre-scores every APP-000001 … APP-N at boot time using
# the score executor so live requests hit only the dict.
# Done in background so the server accepts traffic immediately.
# =========================================================
def _warm_cache_worker(start_i: int, end_i: int):
    for i in range(start_i, end_i):
        aid = f"APP-{i+1:06d}"
        try:
            generate_risk_score(aid)
        except Exception:
            pass

def _start_cache_warmup():
    n        = min(len(applications_df), TOTAL_APPLICATIONS)
    workers  = 16
    chunk    = math.ceil(n / workers)
    futures  = []
    for w in range(workers):
        s = w * chunk
        e = min(s + chunk, n)
        if s < n:
            futures.append(_score_executor.submit(_warm_cache_worker, s, e))

    def _log_done(fs):
        for f in as_completed(fs): pass
        print(f"✅ Cache warm-up complete — {n} applications pre-scored")

    threading.Thread(target=_log_done, args=(futures,), daemon=True).start()

_start_cache_warmup()   # fire immediately, don't block startup

# =========================================================
# BACKGROUND HELPERS
# =========================================================
def _audit_worker(payload: dict):
    conn = None
    try:
        conn   = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_trail
            (application_id, decision, decision_notes, applicant_name, timestamp)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (payload["application_id"], payload["decision"],
               payload["notes"], payload["applicant_name"]))
        conn.commit()
        cursor.close()
    except Exception as e:
        if conn: conn.rollback()
        print(f"[AUDIT ERROR] {e}")
    finally:
        if conn: db_pool.putconn(conn)

def _dispatch_audit(payload: dict):
    _audit_executor.submit(_audit_worker, payload)

def _write_log(entry: dict):
    try:
        with open("model_predictions.log", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

def _dispatch_log(entry: dict):
    _log_executor.submit(_write_log, entry)

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
        "risk_cache_size":    len(_risk_cache),
        "cibil_source":       "computed_from_foir_income_lti_loans_employment",
    }

# =========================================================
# SCORE SINGLE  — cache hit returns in <1 ms
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):
    start_time = time.time()
    try:
        result     = generate_risk_score(req.application_id)
        latency_ms = (time.time() - start_time) * 1000
        _dispatch_log({
            "timestamp":      datetime.now().isoformat(),
            "application_id": req.application_id,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"],
            "latency_ms":     round(latency_ms, 2),
            "status":         "success",
        })
        return {
            "application_id": req.application_id,
            "model_loaded":   True,
            "risk_score":     result["risk_score"],
            "risk_tier":      result["risk_tier"],
            "features_used":  len(MODEL_FEATURES),
            "latency_ms":     round(latency_ms, 2),
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        _dispatch_log({
            "timestamp": datetime.now().isoformat(),
            "application_id": req.application_id,
            "latency_ms": round(latency_ms, 2),
            "status": "error", "error": str(e),
        })
        return {"application_id": req.application_id, "model_loaded": False, "error": str(e)}

# =========================================================
# SCORE BATCH  — parallel via persistent pool
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    futures = {_score_executor.submit(generate_risk_score, aid): aid
               for aid in req.application_ids}
    results = []
    for fut, aid in futures.items():
        r = fut.result()
        results.append({"application_id": aid,
                         "risk_score": r["risk_score"],
                         "risk_tier":  r["risk_tier"]})
    return {"total_applications": len(results), "results": results}

# =========================================================
# APPLICATIONS LIST  — all scores from cache, parallel
# =========================================================
@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    try:
        rows, app_ids = [], []
        for i in range(offset, offset + limit):
            row = applications_df.iloc[i % len(applications_df)].copy()
            aid = f"APP-{i+1:06d}"
            row["application_id"] = aid
            rows.append(row)
            app_ids.append(aid)

        futures   = {_score_executor.submit(generate_risk_score, aid): aid for aid in app_ids}
        score_map = {aid: fut.result() for fut, aid in futures.items()}

        applications = []
        for row, aid in zip(rows, app_ids):
            r              = score_map[aid]
            risk_score     = r["risk_score"]
            risk_tier      = r["risk_tier"]
            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))
            applications.append({
                "application_id":     aid,
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         risk_score,
                "risk_tier":          risk_tier,
                "cibil_score":        _lookup_cibil(aid),
                "credit_score":       get_ml_credit_score(risk_score),
                "application_status": get_status(risk_tier),
            })
        return {"total": TOTAL_APPLICATIONS, "applications": applications}
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# APPLICATION DETAIL  — O(1) lookup + cached score/cibil
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):
    try:
        row, canonical_id = _lookup_row(application_id)
        if row is None:
            return {"error": "Application not found"}

        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))
        foir           = round((monthly_emi / monthly_income) * 100, 2) if monthly_income > 0 else 0

        score_data   = generate_risk_score(application_id)
        risk_score   = score_data["risk_score"]
        risk_tier    = score_data["risk_tier"]
        cibil_score  = _lookup_cibil(application_id)
        credit_score = get_ml_credit_score(risk_score)

        application_status = safe_str(row.get("application_status", row.get("status", "")))
        if not application_status:
            application_status = (
                "Rejected" if risk_score >= 0.75 else
                "Pending"  if risk_score >= 0.45 else
                "Approved"
            )

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
            "date_applied":       safe_str(row.get("application_date", row.get("date_applied", ""))),
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# =========================================================
# AUDIT HISTORY  — DB read only; auto-insert is async
# =========================================================
@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):
    try:
        clean_id = str(application_id).strip().upper()
        row, _   = _lookup_row(application_id)
        csv_name = safe_str(row.get("applicant_name", "Unknown Applicant")) if row is not None else "Unknown Applicant"

        conn   = get_db_connection()
        cursor = conn.cursor()
        # Recommended: CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_app_id
        #              ON audit_trail (UPPER(TRIM(application_id)));
        cursor.execute("""
            SELECT audit_id, decision, decision_notes, timestamp, applicant_name
            FROM audit_trail
            WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC
        """, (clean_id,))
        rows = cursor.fetchall()
        cursor.close()
        db_pool.putconn(conn)

        if not rows:
            score_data  = generate_risk_score(application_id)
            risk_tier   = score_data["risk_tier"]
            decision    = {"Low": "APPROVE", "Medium": "REVIEW", "High": "REJECT"}.get(risk_tier, "REVIEW")
            status_note = (
                f"Credit profile {risk_tier.lower()} risk — "
                f"auto decision based on model score {score_data['risk_score']}"
            )
            app_date = safe_str(row.get("application_date", "")) if row is not None else ""

            # fire-and-forget
            _dispatch_audit({"application_id": application_id, "decision": decision,
                              "notes": status_note, "applicant_name": csv_name})
            return {"history": [{
                "audit_id":       "queued",
                "decision":       decision,
                "notes":          status_note,
                "timestamp":      app_date,
                "applicant_name": csv_name,
            }]}

        history = []
        for r in rows:
            db_val = r[4]
            db_str = str(db_val).strip() if db_val is not None else ""
            name   = csv_name if (not db_val or db_str.lower() in ["", "none", "null"]) else safe_str(db_val)
            history.append({
                "audit_id":       r[0],
                "decision":       r[1],
                "notes":          r[2],
                "timestamp":      r[3].isoformat() if r[3] else None,
                "applicant_name": name,
            })
        return {"history": history}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# PROCESS DECISION
# UPDATE is synchronous. Audit INSERT is fire-and-forget.
# Response returns the instant UPDATE commits (~5-10ms).
# =========================================================
@app.post("/api/applications/{application_id}/process-decision")
def process_decision(application_id: str, req: DecisionRequest):
    decision_map = {
        "APPROVE": "APPROVE", "APPROVED": "APPROVE",
        "REJECT":  "REJECT",  "REJECTED": "REJECT",
        "REVIEW":  "REVIEW",
    }
    decision = str(req.decision).strip().upper()
    if decision not in decision_map:
        return {"status": "failed",
                "error": "Invalid decision. Allowed: APPROVE, REJECT, REVIEW"}
    decision  = decision_map[decision]
    notes     = req.notes or ""
    search_id = str(application_id).strip().upper()

    row, _ = _lookup_row(application_id)
    if row is None:
        return JSONResponse(status_code=404,
                            content={"status": "failed",
                                     "error": f"Application {application_id} not found"})

    applicant_name = safe_str(row.get("applicant_name", "Unknown Applicant"))

    status_sql = {
        "APPROVE": "SET application_status='approved', updated_at=CURRENT_TIMESTAMP",
        "REJECT":  "SET application_status='rejected', updated_at=CURRENT_TIMESTAMP",
        "REVIEW":  "SET application_status='under_review', assigned_reviewer='TEAM_LEAD', updated_at=CURRENT_TIMESTAMP",
    }
    notification_map = {
        "APPROVE": "approval_email",
        "REJECT":  "rejection_email",
        "REVIEW":  "internal_review_notification",
    }

    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE applications {status_sql[decision]} WHERE UPPER(TRIM(application_id))=%s",
            (search_id,)
        )
        conn.commit()
        cursor.close()
        db_pool.putconn(conn)
        conn = None
    except Exception as e:
        if conn:
            conn.rollback()
            try: cursor.close()
            except: pass
            db_pool.putconn(conn)
        return JSONResponse(status_code=500,
                            content={"status": "failed",
                                     "application_id": application_id, "error": str(e)})

    _dispatch_audit({"application_id": application_id, "decision": decision,
                     "notes": notes, "applicant_name": applicant_name})

    return {
        "application_id":    application_id,
        "applicant_name":    applicant_name,
        "audit_id":          "queued",
        "status":            decision.lower(),
        "next_action":       notification_map[decision],
        "notification_sent": True,
        "message":           "Decision processed successfully",
    }

# =========================================================
# PORTFOLIO SUMMARY  — vectorised numpy, no per-row loops
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():
    start = time.time()
    try:
        df = applications_df

        def get_col(name):
            return pd.to_numeric(df[name], errors="coerce").fillna(0) if name in df.columns \
                   else pd.Series(0.0, index=df.index)

        monthly_income     = get_col("monthly_income")
        num_existing_loans = get_col("num_existing_loans")
        employment_years   = get_col("employment_years")
        foir               = get_col("foir")
        loan_to_income     = get_col("loan_to_income_ratio")

        score = pd.Series(750.0, index=df.index)
        score += np.select([foir<=30,foir<=40,foir<=50,foir<=60],[40,10,-20,-60],default=-100)
        score += np.select([monthly_income>=100000,monthly_income>=75000,
                            monthly_income>=50000,monthly_income>=30000],[50,35,20,5],default=-20)
        score += np.select([loan_to_income<=2,loan_to_income<=4,loan_to_income<=6],[30,10,-20],default=-50)
        extra  = np.where(num_existing_loans>2,-30*(num_existing_loans-2),0)
        score += np.select([num_existing_loans==0,num_existing_loans==1,
                            num_existing_loans==2],[20,5,-15],default=extra)
        score += np.select([employment_years>=10,employment_years>=5,
                            employment_years>=3,employment_years>=1],[40,25,10,-5],default=-25)
        score  = score.clip(300,900).astype(int)

        low    = int((score >= 750).sum())
        medium = int(((score >= 650) & (score < 750)).sum())
        high   = int((score < 650).sum())
        elapsed = round(time.time() - start, 2)
        print(f"✅ Portfolio Summary: high={high}, medium={medium}, low={low}, time={elapsed}s")
        return {
            "total_applications":     TOTAL_APPLICATIONS,
            "high": high, "medium": medium, "low": low,
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
