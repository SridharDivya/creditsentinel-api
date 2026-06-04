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

from typing import List, Optional

from feature_engine import compute_features

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
# PRINT ALL COLUMNS + SAMPLE VALUES AT STARTUP
# =========================================================
print(f"\n{'='*60}")
print(f"ALL CSV COLUMNS ({len(applications_df.columns)} total):")
for i, col in enumerate(applications_df.columns):
    sample = applications_df[col].dropna().head(3).tolist()
    print(f"  [{i:02d}] '{col}'  →  {sample}")
print(f"{'='*60}\n")

# =========================================================
# SMART CIBIL COLUMN DETECTION  (3-pass strategy)
# =========================================================
CIBIL_COLUMN = None

# --- Pass 1: exact name match (case-sensitive) ---
EXACT_CANDIDATES = [
    "cibil_score", "credit_score", "cibil", "bureau_score",
    "creditScore", "CIBIL Score", "CIBIL_score", "Credit Score",
    "bureau_credit_score", "fico_score", "credit_rating",
    "CreditScore", "CREDIT_SCORE", "CIBIL", "BureauScore",
    "cibil score", "Credit_Score", "bureau_cibil_score",
    "cibil_rating", "credit_bureau_score", "cb_score",
    "risk_score_bureau", "external_score", "bureau_rating"
]
for candidate in EXACT_CANDIDATES:
    if candidate in applications_df.columns:
        CIBIL_COLUMN = candidate
        print(f"✅ CIBIL column (Pass 1 exact): '{CIBIL_COLUMN}'")
        break

# --- Pass 2: case-insensitive keyword scan ---
if CIBIL_COLUMN is None:
    CIBIL_KEYWORDS = ["cibil", "bureau", "fico", "credit_score",
                      "creditscore", "credit_rating", "creditrating"]
    for col in applications_df.columns:
        col_norm = col.lower().replace(" ", "_").replace("-", "_")
        for kw in CIBIL_KEYWORDS:
            if kw in col_norm:
                CIBIL_COLUMN = col
                print(f"✅ CIBIL column (Pass 2 fuzzy keyword '{kw}'): '{CIBIL_COLUMN}'")
                break
        if CIBIL_COLUMN:
            break

# --- Pass 3: value-range heuristic ---
# Find any numeric column where the majority of values fall in 300–900
# (typical CIBIL/bureau score range) and it's NOT a loan amount or income.
# This works regardless of what the column is named.
if CIBIL_COLUMN is None:
    print("⚠️  Passes 1 & 2 failed — trying value-range heuristic (Pass 3)...")
    EXCLUDE_KEYWORDS = ["income", "amount", "emi", "loan", "salary",
                        "age", "tenure", "month", "year", "id"]
    candidates_pass3 = []
    for col in applications_df.columns:
        col_lower = col.lower()
        # skip obviously non-score columns
        if any(kw in col_lower for kw in EXCLUDE_KEYWORDS):
            continue
        series = pd.to_numeric(applications_df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        # Check: ≥70% of values in 300–900 AND median in that range
        in_range = ((series >= 300) & (series <= 900)).sum()
        pct_in_range = in_range / len(series)
        median_val = series.median()
        if pct_in_range >= 0.70 and 300 <= median_val <= 900:
            candidates_pass3.append((col, pct_in_range, float(median_val)))
            print(f"   Pass 3 candidate: '{col}'  pct_in_range={pct_in_range:.2f}  median={median_val:.0f}")

    if candidates_pass3:
        # Pick the column with the highest % of values in 300–900
        CIBIL_COLUMN = sorted(candidates_pass3, key=lambda x: x[1], reverse=True)[0][0]
        print(f"✅ CIBIL column (Pass 3 value-range heuristic): '{CIBIL_COLUMN}'")

if CIBIL_COLUMN is None:
    print("❌ CIBIL column NOT found by any method.")
    print("   → Deploy this code, then call GET /api/debug/columns")
    print("     to see every column name + sample values,")
    print("     then tell us the exact column name to hard-code it.")
else:
    sample_vals = applications_df[CIBIL_COLUMN].dropna().head(5).tolist()
    print(f"   Confirmed CIBIL sample values: {sample_vals}")

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

def get_credit_score_from_risk(risk_score: float) -> int:
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
# SCORE FUNCTIONS
# cibil_score  = real bureau score from CSV (e.g. 573, 706)
# credit_score = ML-derived 300–900 from risk model
# =========================================================
def get_raw_cibil_score(row) -> int:
    """Read actual CIBIL/bureau score directly from the CSV row."""
    if CIBIL_COLUMN is None:
        return 0
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    val = row_dict.get(CIBIL_COLUMN, None)
    if val is None:
        return 0
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0
    try:
        score = int(float(val))
        return max(score, 0)
    except (ValueError, TypeError):
        return 0

def get_ml_credit_score(risk_score: float) -> int:
    """ML-derived credit score (300–900). Always has a valid value."""
    return get_credit_score_from_risk(risk_score)

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

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status":                    "ok",
        "model_loaded":              True,
        "total_applications":        len(applications_df),
        "cibil_column_detected":     CIBIL_COLUMN or "NOT_FOUND",
        "total_csv_columns":         len(applications_df.columns),
        "tip":                       "If cibil_column_detected is NOT_FOUND, call /api/debug/columns"
    }

# =========================================================
# DEBUG: ALL COLUMNS — paste the output here if CIBIL is still 0
# =========================================================
@app.get("/api/debug/columns")
def debug_columns():
    """
    Returns every column name in loan_applications.csv with sample values.
    Call this when cibil_score is 0 — find the column that has values like
    573, 706, 648, etc., and share it so we can hard-code the correct name.
    """
    col_info = {}
    for col in applications_df.columns:
        series  = applications_df[col]
        sample  = series.dropna().head(5).tolist()
        numeric = pd.to_numeric(series, errors="coerce")
        in_range = int(((numeric >= 300) & (numeric <= 900)).sum())
        col_info[col] = {
            "sample_values":      sample,
            "dtype":              str(series.dtype),
            "null_count":         int(series.isna().sum()),
            "values_in_300_900":  in_range,        # non-zero = likely a credit score column
        }
    return {
        "detected_cibil_column": CIBIL_COLUMN or "NOT_FOUND",
        "columns":               col_info
    }

# =========================================================
# DEBUG: RAW ROW for one application
# =========================================================
@app.get("/api/debug/application/{application_id}")
def debug_application(application_id: str):
    matched = applications_df[
        applications_df["application_id"].astype(str) == str(application_id)
    ]
    if len(matched) == 0:
        return {"error": "Application not found"}
    row      = matched.iloc[0]
    raw_data = {}
    for col, val in row.to_dict().items():
        raw_data[col] = None if (isinstance(val, float) and math.isnan(val)) else val
    return {
        "application_id":          application_id,
        "detected_cibil_column":   CIBIL_COLUMN or "NOT_FOUND",
        "cibil_value_from_csv":    raw_data.get(CIBIL_COLUMN) if CIBIL_COLUMN else "column_not_found",
        "raw_row":                 raw_data
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
            cibil_score    = get_raw_cibil_score(row)
            credit_score   = get_ml_credit_score(risk_score)

            applications.append({
                "application_id":     app_id,
                "applicant_name":     safe_str(row.get("applicant_name", "")),
                "foir":               get_foir(monthly_income, monthly_emi),
                "monthly_income":     monthly_income,
                "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
                "risk_score":         risk_score,
                "risk_tier":          risk_tier,
                "cibil_score":        cibil_score,   # real CSV bureau score
                "credit_score":       credit_score,  # ML-derived 300–900
                "application_status": get_status(risk_tier)
            })

        return {"total": len(applications_df), "applications": applications}

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
        cibil_score  = get_raw_cibil_score(row)        # real CSV bureau score
        credit_score = get_ml_credit_score(risk_score) # ML-derived 300–900

        application_status = safe_str(row.get("application_status", row.get("status", "")))
        if application_status == "":
            if risk_score >= 0.75:   application_status = "Rejected"
            elif risk_score >= 0.45: application_status = "Pending"
            else:                    application_status = "Approved"

        print(
            f"[DETAIL] id={application_id} | "
            f"col='{CIBIL_COLUMN}' | "
            f"cibil={cibil_score} | "
            f"credit={credit_score} | "
            f"risk={risk_score}"
        )

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
        sample_df = applications_df.sample(n=min(500, len(applications_df)), random_state=42)

        for _, row in sample_df.iterrows():
            app_id = safe_str(row.get("application_id", ""))
            tier   = generate_risk_score(app_id)["risk_tier"]
            if tier == "High":     high   += 1
            elif tier == "Medium": medium += 1
            else:                  low    += 1

        total = len(applications_df)
        scale = total / len(sample_df)

        return {
            "total_applications": total,
            "high":   round(high   * scale),
            "medium": round(medium * scale),
            "low":    round(low    * scale)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
