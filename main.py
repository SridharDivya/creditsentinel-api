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
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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
# DB STARTUP AUTOMATION INDEXES
# =========================================================
def _ensure_db_indexes():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_app_id_upper
            ON audit_trail (UPPER(TRIM(application_id)));
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_apps_app_id_upper
            ON applications (UPPER(TRIM(application_id)));
        """)
        conn.commit()
        cursor.close()
        print("✅ DB Performance Indexes Confirmed")
    except Exception as e:
        print(f"[INDEX WARN] Could not initialize indexes: {e}")
        if conn: conn.rollback()
    finally:
        if conn: db_pool.putconn(conn)

_ensure_db_indexes()

# =========================================================
# THREAD EXECUTORS
# =========================================================
_audit_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="audit")
_worker_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="pipeline")

def _audit_worker(payload: dict):
    conn = None
    try:
        conn   = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_trail
                (application_id, decision, decision_notes,
                 applicant_name, analyst_name, timestamp)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING audit_id
        """, (
            payload["application_id"],
            payload["decision"],
            payload["notes"],
            payload["applicant_name"],
            payload.get("analyst_name", ""),
        ))
        audit_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return audit_id
    except Exception as e:
        if conn: conn.rollback()
        print(f"[AUDIT WORKER ERROR] {e}")
        return None
    finally:
        if conn: db_pool.putconn(conn)

def fire_and_forget_audit(payload: dict):
    _audit_executor.submit(_audit_worker, payload)

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

def fire_and_forget_email(recipient: str, subject: str, body: str):
    _audit_executor.submit(send_email, recipient, subject, body)

# =========================================================
# MODEL FEATURES
# =========================================================
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = list(model.feature_name_())

# =========================================================
# SAFE TYPE HELPERS
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
# SHARED LOGIC HELPERS
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
        if val >= 0: return val
    return 0.0

def get_decision_date(application_id: str) -> str:
    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp) FROM audit_trail WHERE UPPER(TRIM(application_id)) = %s
        """, (application_id.strip().upper(),))
        row = cursor.fetchone()
        cursor.close()
        return row[0].isoformat() if row and row[0] else ""
    except Exception: return ""
    finally:
        if conn: db_pool.putconn(conn)

def get_real_status(application_id: str, risk_tier: str) -> str:
    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT decision FROM audit_trail WHERE UPPER(TRIM(application_id)) = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (application_id.strip().upper(),))
        row = cursor.fetchone()
        cursor.close()
        if row and row[0]:
            return {"APPROVE": "Approved", "REJECT": "Rejected", "REVIEW": "Under Review"}.get(
                str(row[0]).upper(), get_status(risk_tier)
            )
        return get_status(risk_tier)
    except Exception: return get_status(risk_tier)
    finally:
        if conn: db_pool.putconn(conn)

def batch_get_audit_info(app_ids: list) -> dict:
    if not app_ids: return {}
    clean_ids    = [a.strip().upper() for a in app_ids]
    placeholders = ",".join(["%s"] * len(clean_ids))
    conn = None
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT UPPER(TRIM(application_id)) AS app_id, decision, MAX(timestamp) AS latest_ts
            FROM audit_trail WHERE UPPER(TRIM(application_id)) IN ({placeholders})
            GROUP BY UPPER(TRIM(application_id)), decision ORDER BY latest_ts DESC
        """, clean_ids)
        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print(f"[BATCH AUDIT ERROR] {e}")
        return {}
    finally:
        if conn: db_pool.putconn(conn)

    result       = {}
    decision_map = {"APPROVE": "Approved", "REJECT": "Rejected", "REVIEW": "Under Review"}
    for row in rows:
        aid, decision, ts = row
        if aid not in result:
            result[aid] = {
                "status":        decision_map.get(str(decision).upper(), "Pending"),
                "decision_date": ts.isoformat() if ts else "",
            }
    return result

# =========================================================
# RULES RULES & COMPUTED METRICS
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

def get_credit_based_note(decision: str, cibil_score: int, risk_score: float, risk_tier: str) -> str:
    if decision in ("APPROVE", "APPROVED"):
        if cibil_score >= 750:
            return f"Application approved. Excellent credit score of {cibil_score} with low risk profile (score: {risk_score})."
        else:
            return f"Application approved. Credit score {cibil_score} is satisfactory. Risk tier: {risk_tier}."
    elif decision in ("REJECT", "REJECTED"):
        return f"Application rejected. Credit score evaluation threshold mismatch. Risk tier: {risk_tier} (score: {risk_score})."
    return f"Application flagged for manual review. Credit score {cibil_score} requires further assessment."

# =========================================================
# LRU FEATURE PIPELINE CACHE
# =========================================================
@lru_cache(maxsize=4000)
def _cached_features_frozen(application_id: str) -> tuple:
    features = compute_features(application_id)
    return tuple(sorted(features.items()))

_cache_lock = threading.Lock()
def _get_cached_features(application_id: str) -> dict:
    with _cache_lock:
        return dict(_cached_features_frozen(application_id))

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

def generate_risk_scores_batch(application_ids: list) -> dict:
    if not application_ids: return {}
    try:
        features_dicts = list(_worker_executor.map(_get_cached_features, application_ids))
        rows = []
        for features_dict in features_dicts:
            rows.append({f: features_dict.get(f, 0) for f in MODEL_FEATURES})

        features_df = (
            pd.DataFrame(rows, index=application_ids)[MODEL_FEATURES]
            .fillna(0).replace([np.inf, -np.inf], 0).astype(float)
        )
        proba  = model.predict_proba(features_df)[:, 1]
        return {app_id: {"risk_score": round(float(s), 4), "risk_tier": get_risk_tier(float(s))} for app_id, s in zip(application_ids, proba)}
    except Exception:
        return {app_id: {"risk_score": 0.0, "risk_tier": "Low"} for app_id in application_ids}

# =========================================================
# SCHEMAS
# =========================================================
class ScoreRequest(BaseModel):
    application_id: str

class BatchScoreRequest(BaseModel):
    application_ids: List[str]

class DecisionRequest(BaseModel):
    decision:     str
    notes:        Optional[str] = ""
    analyst_name: str

# =========================================================
# HTTP API ENDPOINTS
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok", "model_loaded": True, "total_applications": TOTAL_APPLICATIONS,
        "optimizations": ["batch_db_lookup", "lru_feature_cache", "batch_model_inference", "parallel_worker_pipelines"]
    }

@app.post("/api/score")
def score_application(req: ScoreRequest):
    start_time = time.time()
    try:
        res = generate_risk_score(req.application_id)
        latency = (time.time() - start_time) * 1000
        return {"application_id": req.application_id, "model_loaded": True, "risk_score": res["risk_score"], "risk_tier": res["risk_tier"], "features_used": len(MODEL_FEATURES), "latency_ms": round(latency, 2)}
    except Exception as e:
        return {"application_id": req.application_id, "model_loaded": False, "error": str(e)}

@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):
    score_map = generate_risk_scores_batch(req.application_ids)
    return {"total_applications": len(score_map), "results": [{"application_id": k, **v} for k, v in score_map.items()]}

@app.get("/api/applications")
def get_applications(limit: int = 10, offset: int = 0):
    request_start = time.time()
    try:
        rows_list = []
        for i in range(offset, offset + limit):
            row = applications_df.iloc[i % len(applications_df)].copy()
            row["application_id"] = f"APP-{i+1:06d}"
            rows_list.append(row)
        subset = pd.DataFrame(rows_list)
        app_ids = [safe_str(row.get("application_id", "")) for _, row in subset.iterrows()]

        score_map = generate_risk_scores_batch(app_ids)
        audit_map = batch_get_audit_info(app_ids)

        applications = []
        for _, row in subset.iterrows():
            app_id         = safe_str(row.get("application_id", ""))
            res            = score_map.get(app_id, {"risk_score": 0.0, "risk_tier": "Low"})
            audit_info     = audit_map.get(app_id.strip().upper(), {})
            monthly_income = safe_float(row.get("monthly_income", 0))
            monthly_emi    = get_emi_from_row(row)

            applications.append({
                "application_id":     app_id,
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         res["risk_score"],
                "risk_tier":          res["risk_tier"],
                "cibil_score":        compute_cibil_score(row),
                "credit_score":       get_ml_credit_score(res["risk_score"]),
                "application_status": audit_info.get("status", get_status(res["risk_tier"])),
                "created_at":         safe_str(row.get("created_at", row.get("application_date", ""))),
                "decision_date":      audit_info.get("decision_date", ""),
            })

        print(f"[PROFILE] Applications page list metrics handled in {((time.time() - request_start) * 1000):.2f}ms")
        return {"total": TOTAL_APPLICATIONS, "applications": applications}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):
    try:
        matched = applications_df[applications_df["application_id"].astype(str) == str(application_id)]
        if len(matched) == 0:
            numeric = int(str(application_id).split("-")[-1]) - 1
            row = applications_df.iloc[numeric % len(applications_df)].copy()
            row["application_id"] = application_id
        else:
            row = matched.iloc[0]
            
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = get_emi_from_row(row)
        score_data     = generate_risk_score(application_id)

        return {
            "application_id":     application_id,
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", row.get("loan_amount", 0))),
            "foir":               get_foir(monthly_income, monthly_emi),
            "cibil_score":        compute_cibil_score(row),
            "credit_score":       get_ml_credit_score(score_data["risk_score"]),
            "risk_score":         score_data["risk_score"],
            "risk_tier":          score_data["risk_tier"],
            "application_status": get_real_status(application_id, score_data["risk_tier"]),
            "date_applied":       safe_str(row.get("application_date", row.get("date_applied", ""))),
            "decision_date":      get_decision_date(application_id),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/applications/{application_id}/history")
def get_decision_history(application_id: str):
    history_start = time.time()
    try:
        clean_id = str(application_id).strip().upper()
        matched  = applications_df[applications_df["application_id"].astype(str).str.strip().str.upper() == clean_id]
        
        csv_row = matched.iloc[0] if len(matched) > 0 else None
        csv_applicant_name = safe_str(csv_row["applicant_name"]) if csv_row is not None else "Unknown Applicant"
        csv_application_date = safe_str(csv_row.get("application_date", "")) if csv_row is not None else ""

        score_data  = generate_risk_score(application_id)
        cibil_score = compute_cibil_score(csv_row) if csv_row is not None else 650

        email_to = MAIL_TEST_RECIPIENT
        if csv_row is not None:
            for col in ["email", "email_address", "applicant_email"]:
                v = safe_str(csv_row.get(col, ""))
                if v.strip():
                    email_to = v.strip()
                    break

        conn = None
        rows = []
        try:
            conn   = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT audit_id, decision, decision_notes, timestamp, analyst_name
                FROM audit_trail WHERE UPPER(TRIM(application_id)) = %s ORDER BY timestamp DESC
            """, (clean_id,))
            rows = cursor.fetchall()
            cursor.close()
        finally:
            if conn: db_pool.putconn(conn)

        if not rows:
            decision = {"Low": "APPROVE", "Medium": "REVIEW", "High": "REJECT"}.get(score_data["risk_tier"], "REVIEW")
            status_note = get_credit_based_note(decision, cibil_score, score_data["risk_score"], score_data["risk_tier"])
            
            real_audit_id = _audit_worker({
                "application_id": application_id, "decision": decision, "notes": status_note,
                "applicant_name": csv_applicant_name, "analyst_name": "SYSTEM"
            })
            
            fire_and_forget_email(email_to, f"History Report – {application_id}", f"Applicant: {csv_applicant_name}\nDecision: {decision}")
            return {
                "history": [{
                    "audit_id": real_audit_id, "decision": decision, "notes": status_note,
                    "timestamp": csv_application_date or datetime.now().isoformat(),
                    "applicant_name": csv_applicant_name, "analyst_name": "SYSTEM"
                }],
                "email_report": True, "email_to": email_to, "latency_ms": round((time.time() - history_start) * 1000, 2)
            }

        history = []
        for row in rows:
            history.append({
                "audit_id": row[0], "decision": safe_str(row[1]), "notes": safe_str(row[2]),
                "timestamp": row[3].isoformat() if row[3] else None, "applicant_name": csv_applicant_name,
                "analyst_name": safe_str(row[4]) if row[4] else "SYSTEM"
            })

        fire_and_forget_email(email_to, f"History Report – {application_id}", f"Records compiled: {len(history)}")
        return {"history": history, "email_report": True, "email_to": email_to, "latency_ms": round((time.time() - history_start) * 1000, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/applications/{application_id}/process-decision")
def process_decision(application_id: str, req: DecisionRequest):
    decision_map = {"APPROVE": "APPROVED", "APPROVED": "APPROVED", "REJECT": "REJECTED", "REJECTED": "REJECTED", "REVIEW": "UNDER_REVIEW"}
    decision = str(req.decision).strip().upper()
    if decision not in decision_map:
        return {"status": "failed", "error": "Invalid decision constraint profile match"}
    
    target_status = decision_map[decision].lower()
    search_id = str(application_id).strip().upper()
    
    matched = applications_df[applications_df["application_id"].astype(str).str.strip().str.upper() == search_id]
    real_applicant_name = safe_str(matched.iloc[0].get("applicant_name", "Unknown Applicant")) if len(matched) > 0 else "Unknown Applicant"
    
    recipient_email = MAIL_TEST_RECIPIENT
    if len(matched) > 0:
        for col in ["email", "email_address", "applicant_email"]:
            val = safe_str(matched.iloc[0].get(col, ""))
            if val.strip():
                recipient_email = val.strip()
                break

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE applications SET application_status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE UPPER(TRIM(application_id)) = %s
        """, (target_status, search_id))
        conn.commit()
        cursor.close()
    except Exception as e:
        if conn: conn.rollback()
        return JSONResponse(status_code=500, content={"status": "failed", "error": f"Database operation failed: {str(e)}"})
    finally:
        if conn: db_pool.putconn(conn)

    fire_and_forget_email(
        recipient_email, 
        f"Loan Application State Update", 
        f"Hello {real_applicant_name},\nYour loan application {application_id} tracking status changed to: {target_status}."
    )

    fire_and_forget_audit({
        "application_id": application_id, "decision": decision,
        "notes": req.notes or "", "applicant_name": real_applicant_name, "analyst_name": req.analyst_name
    })
    
    return {"status": "success", "application_id": application_id, "message": "Decision committed instantly"}

@app.get("/api/portfolio/summary")
def portfolio_summary():
    start = time.time()
    try:
        df = applications_df
        def get_col(name):
            return pd.to_numeric(df[name], errors="coerce").fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)

        monthly_income, num_existing_loans = get_col("monthly_income"), get_col("num_existing_loans")
        employment_years, foir, loan_to_income = get_col("employment_years"), get_col("foir"), get_col("loan_to_income_ratio")

        score = pd.Series(750.0, index=df.index)
        score += np.select([foir<=30, foir<=40, foir<=50, foir<=60], [40,10,-20,-60], default=-100)
        score += np.select([monthly_income>=100000, monthly_income>=75000, monthly_income>=50000,  monthly_income>=30000], [50,35,20,5], default=-20)
        score += np.select([loan_to_income<=2, loan_to_income<=4, loan_to_income<=6], [30,10,-20], default=-50)
        score += np.select([num_existing_loans==0, num_existing_loans==1, num_existing_loans==2], [20,5,-15], default=np.where(num_existing_loans>2, -30*(num_existing_loans-2), 0))
        score += np.select([employment_years>=10, employment_years>=5, employment_years>=3,  employment_years>=1], [40,25,10,-5], default=-25)
        score = score.clip(300, 900).astype(int)

        return {"total_applications": TOTAL_APPLICATIONS, "high": int((score < 650).sum()), "medium": int(((score >= 650) & (score < 750)).sum()), "low": int((score >= 750).sum()), "execution_time_seconds": round(time.time() - start, 3)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=4)
