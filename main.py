 =========================================================
# CREDITSENTINEL FASTAPI - RENDER VERSION
# =========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import os
import math
import traceback

from typing import List

from feature_engine import compute_features

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print("BASE DIRECTORY:", BASE_DIR)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "creditsentinel_model_v1.pkl"
    )
)
from config import Config
model = joblib.load(Config.MODEL_PATH)

print("✅ Real model loaded")
print(type(model))
# 

print("✅ Model Loaded")

# =========================================================
# LOAD CSV FILES
# =========================================================

bank_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "bank_statements.csv"
    )
)

bureau_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "bureau_data.csv"
    )
)

gst_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "gst_filings.csv"
    )
)

applications_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "loan_applications.csv"
    )
)

print("✅ CSV Files Loaded")

print("Applications Shape:")
print(applications_df.shape)

print("Applications Columns:")
print(applications_df.columns.tolist())

# =========================================================
# SAFE HELPER FUNCTIONS
# =========================================================

def safe_float(val, default=0.0):

    try:

        result = float(val)

        if math.isnan(result) or math.isinf(result):

            return default

        return result

    except:

        return default


def safe_int(val, default=0):

    try:

        result = float(val)

        if math.isnan(result) or math.isinf(result):

            return default

        return int(result)

    except:

        return default


def safe_str(val, default=""):

    try:

        if val is None:

            return default

        if isinstance(val, float):

            if math.isnan(val) or math.isinf(val):

                return default

        return str(val)

    except:

        return default

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="CreditSentinel API",
    description="Loan Risk Scoring API",
    version="1.0"
)

# =========================================================
# ENABLE CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST MODELS
# =========================================================

class ScoreRequest(BaseModel):

    application_id: str


class BatchScoreRequest(BaseModel):

    application_ids: List[str]


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",
        "model_loaded": True,
        "applications": len(applications_df)
    }

# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================

@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        if not req.application_id.strip():

            return {

                "error":
                "Invalid application ID"
            }

        # =================================================
        # COMPUTE FEATURES
        # =================================================

        features_dict = compute_features(
            req.application_id
        )

        # =================================================
        # MODEL FEATURES
        # =================================================

        if hasattr(model, "feature_names_in_"):

            model_features = list(
                model.feature_names_in_
            )

        elif hasattr(model, "feature_name_"):

            model_features = list(
                model.feature_name_
            )

        else:

            return {

                "application_id":
                req.application_id,

                "error":
                "Model feature names not found"
            }

        # =================================================
        # FILTER FEATURES
        # =================================================

        features_filtered = {

            feature: features_dict.get(feature, 0)

            for feature in model_features
        }

        # =================================================
        # DATAFRAME
        # =================================================

        features_df = pd.DataFrame(
            [features_filtered]
        )

        features_df = features_df[
            model_features
        ]

        # =================================================
        # CLEAN VALUES
        # =================================================

        features_df = features_df.fillna(0)

        features_df = features_df.replace(
            [np.inf, -np.inf],
            0
        )

        # =================================================
        # NUMERIC CONVERSION
        # =================================================

        for col in features_df.columns:

            try:

                features_df[col] = pd.to_numeric(
                    features_df[col],
                    errors="coerce"
                ).fillna(0)

            except:

                pass

        # =================================================
        # PREDICT
        # =================================================

        prediction = model.predict_proba(
            features_df
        )

        risk_score = round(
            float(prediction[:,1][0]),
            4
        )

        # =================================================
        # RISK TIER
        # =================================================

        if risk_score < 0.4:

            tier = "Low"

        elif risk_score < 0.65:

            tier = "Medium"

        else:

            tier = "High"

        # =================================================
        # RESPONSE
        # =================================================

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            True,

            "risk_score":
            risk_score,

            "risk_tier":
            tier,

            "features_used":
            len(features_df.columns)
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            False,

            "error":
            str(e)
        }

# =========================================================
# SCORE BATCH
# =========================================================

@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):

    results = []

    for app_id in req.application_ids:

        try:

            features_dict = compute_features(
                app_id
            )

            if hasattr(model, "feature_names_in_"):

                model_features = list(
                    model.feature_names_in_
                )

            else:

                model_features = list(
                    model.feature_name_
                )

            features_filtered = {

                feature: features_dict.get(feature, 0)

                for feature in model_features
            }

            features_df = pd.DataFrame(
                [features_filtered]
            )

            features_df = features_df[
                model_features
            ]

            features_df = features_df.fillna(0)

            features_df = features_df.replace(
                [np.inf, -np.inf],
                0
            )

            for col in features_df.columns:

                try:

                    features_df[col] = pd.to_numeric(
                        features_df[col],
                        errors="coerce"
                    ).fillna(0)

                except:

                    pass

            prediction = model.predict_proba(
                features_df
            )

            risk_score = round(
                float(prediction[:,1][0]),
                4
            )

            if risk_score < 0.3:

                tier = "Low"

            elif risk_score < 0.6:

                tier = "Medium"

            else:

                tier = "High"

            results.append({

                "application_id":
                app_id,

                "model_loaded":
                True,

                "risk_score":
                risk_score,

                "risk_tier":
                tier
            })

        except Exception as e:

            results.append({

                "application_id":
                app_id,

                "model_loaded":
                False,

                "error":
                str(e)
            })

    return {

        "total_applications":
        len(req.application_ids),

        "results":
        results
    }

# =========================================================
# APPLICATION LIST
# =========================================================

@app.get("/api/applications")
def get_applications(limit: int = 100):
    applications = []
    for _, row in applications_df.head(limit).iterrows():
        app_id = safe_str(row.get("application_id", ""))

        score_result = score_application(
            ScoreRequest(application_id=app_id)
        )

        risk_score = safe_float(score_result.get("risk_score", 0))
        risk_tier  = safe_str(score_result.get("risk_tier", "Unknown"))

        credit_score = int(300 + (1 - risk_score) * 600)

        status_map = {
            "Low":    "Approved",
            "Medium": "Under Review",
            "High":   "Rejected"
        }
        application_status = status_map.get(risk_tier, "Pending")

        # ── FOIR Calculation ───────────────────────────────
        monthly_income = safe_float(row.get("monthly_income", 0))
        monthly_emi    = safe_float(row.get("existing_monthly_emi", 0))

        if monthly_income > 0:
            foir = round((monthly_emi / monthly_income) * 100, 2)
        else:
            foir = 0.0

        applications.append({
            "application_id":     app_id,
            "applicant_name":     safe_str(row.get("applicant_name", "")),
            "foir":               foir,
            "monthly_income":     monthly_income,
            "loan_amount":        safe_float(row.get("requested_loan_amount", 0)),
            "risk_score":         risk_score,
            "risk_tier":          risk_tier,
            "credit_score":       credit_score,
            "application_status": application_status
        })

    return {
        "total": len(applications),
        "applications": applications
    }
# =========================================================
# APPLICATION DETAIL
# =========================================================

@app.get("/api/applications/{application_id}")
def get_application_detail(
    application_id: str
):

    try:

        if "application_id" not in applications_df.columns:

            return {

                "error":
                "application_id column missing"
            }

        matched = applications_df[
            applications_df["application_id"]
            == application_id
        ]

        if len(matched) == 0:

            return {

                "error":
                "Application not found"
            }

        row = matched.iloc[0]

        monthly_income = safe_float(
            row.get("monthly_income", 0)
        )

        monthly_emi = safe_float(
            row.get(
                "existing_monthly_emi",
                0
            )
        )

        if monthly_income > 0:

            foir = round(
                (monthly_emi / monthly_income) * 100,
                2
            )

        else:

            foir = 0

        score_result = score_application(

            ScoreRequest(
                application_id=application_id
            )
        )

        return {

            "application_id":
            safe_str(
                row.get(
                    "application_id",
                    ""
                )
            ),

            "applicant_name":
            safe_str(
                row.get(
                    "applicant_name",
                    ""
                )
            ),

            "monthly_income":
            monthly_income,

            "loan_amount":
            safe_float(
                row.get(
                    "requested_loan_amount",
                    0
                )
            ),

            "foir":
            foir,

            "risk_score":
            safe_float(
                score_result.get(
                    "risk_score",
                    0
                )
            ),

            "risk_tier":
            safe_str(
                score_result.get(
                    "risk_tier",
                    "Low"
                )
            ),

            "credit_score":
            safe_int(
                row.get(
                    "cibil_score",
                    0
                )
            ),

            "application_status":
            safe_str(
                row.get(
                    "application_status",
                    "Pending"
                )
            )
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "error":
            str(e)
        }

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():

    try:

        high_count   = 0
        medium_count = 0
        low_count    = 0

        for _, row in applications_df.iterrows():

            app_id = safe_str(row.get("application_id", ""))

            score_result = score_application(
                ScoreRequest(application_id=app_id)
            )

            risk_tier = safe_str(
                score_result.get("risk_tier", "")
            ).lower()

            if risk_tier == "high":
                high_count += 1
            elif risk_tier == "medium":
                medium_count += 1
            elif risk_tier == "low":
                low_count += 1

        return {
            "total_applications": len(applications_df),
            "high":               high_count,
            "medium":             medium_count,
            "low":                low_count
        }

    except Exception as e:

        print(traceback.format_exc())
        return {"error": str(e)}
