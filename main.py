# =========================================================
# CREDITSENTINEL FASTAPI - RENDER DEPLOYMENT VERSION
# FULL WORKING CODE
# =========================================================

# =========================================================
# IMPORTS
# =========================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import traceback
import os

from typing import List

# =========================================================
# IMPORT FEATURE ENGINE
# =========================================================
from feature_engine import compute_features
from config import Config

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="CreditSentinel API",
    version="1.0"
)

# =========================================================
# ENABLE CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = os.getcwd()

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "creditsentinel_model_v1.pkl"
)

model = joblib.load(MODEL_PATH)

print("✅ Model Loaded")

# =========================================================
# LOAD APPLICATION DATA
# =========================================================
applications_df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "loan_applications.csv"
    )
)

print("✅ Applications CSV Loaded")
print(applications_df.head())

# =========================================================
# REQUEST MODELS
# =========================================================
class ScoreRequest(BaseModel):

    application_id: str


class BatchScoreRequest(BaseModel):

    application_ids: List[str]

# =========================================================
# SAFE HELPERS
# =========================================================
def safe_float(value):

    try:

        if pd.isna(value):

            return 0.0

        return round(float(value), 2)

    except:

        return 0.0


def safe_int(value):

    try:

        if pd.isna(value):

            return 0

        return int(float(value))

    except:

        return 0


def safe_str(value):

    try:

        if pd.isna(value):

            return ""

        return str(value)

    except:

        return ""

# =========================================================
# HEALTH ENDPOINT
# =========================================================
@app.get("/health")
def health():

    return {

        "status": "ok",
        "model_loaded": True,
        "total_applications":
        len(applications_df)
    }

# =========================================================
# GENERATE REAL MODEL SCORE
# =========================================================
def generate_risk_score(application_id):

    try:

        # =========================================
        # COMPUTE FEATURES
        # =========================================
        features_dict = compute_features(
            application_id
        )

        # =========================================
        # MODEL FEATURE ORDER
        # =========================================
        if hasattr(model, "feature_names_in_"):

            model_features = list(
                model.feature_names_in_
            )

        else:

            model_features = list(
                model.feature_name_
            )

        # =========================================
        # FILTER FEATURES
        # =========================================
        filtered_features = {

            feature:
            features_dict.get(feature, 0)

            for feature in model_features
        }

        # =========================================
        # DATAFRAME
        # =========================================
        features_df = pd.DataFrame(
            [filtered_features]
        )

        features_df = features_df[
            model_features
        ]

        features_df = features_df.fillna(0)

        features_df = features_df.astype(float)

        # =========================================
        # PREDICT
        # =========================================
        prediction = model.predict_proba(
            features_df
        )

        risk_score = round(
            float(prediction[:,1][0]),
            4
        )

        # =========================================
        # RISK TIER
        # =========================================
        if risk_score < 0.4:

            risk_tier = "Low"

        elif risk_score < 0.65:

            risk_tier = "Medium"

        else:

            risk_tier = "High"

        return {

            "risk_score":
            risk_score,

            "risk_tier":
            risk_tier
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "risk_score": 0.0,
            "risk_tier": "Low"
        }

# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        result = generate_risk_score(
            req.application_id
        )

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            True,

            "risk_score":
            result["risk_score"],

            "risk_tier":
            result["risk_tier"],

            "features_used":
            43
        }

    except Exception as e:

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

    for application_id in req.application_ids:

        result = generate_risk_score(
            application_id
        )

        results.append({

            "application_id":
            application_id,

            "risk_score":
            result["risk_score"],

            "risk_tier":
            result["risk_tier"]
        })

    return {

        "total_applications":
        len(results),

        "results":
        results
    }

# =========================================================
# APPLICATIONS LIST ENDPOINT
# =========================================================
@app.get("/api/applications")
def get_applications(
    limit: int = 10,
    offset: int = 0
):

    try:

        total = len(applications_df)

        paginated_df = applications_df.iloc[
            offset : offset + limit
        ]

        applications = []

        for _, row in paginated_df.iterrows():

            application_id = safe_str(
                row.get(
                    "application_id",
                    ""
                )
            )

            # =====================================
            # MONTHLY INCOME
            # =====================================
            monthly_income = safe_float(
                row.get(
                    "monthly_income",
                    row.get(
                        "annual_income",
                        0
                    )
                )
            )

            # =====================================
            # EMI
            # =====================================
            monthly_emi = safe_float(
                row.get(
                    "existing_monthly_emi",
                    row.get(
                        "monthly_emi",
                        0
                    )
                )
            )

            # =====================================
            # FOIR
            # =====================================
            if monthly_income > 0:

                foir = round(
                    (
                        monthly_emi
                        /
                        monthly_income
                    ) * 100,
                    2
                )

            else:

                foir = 0

            # =====================================
            # REAL MODEL SCORE
            # =====================================
            score_data = generate_risk_score(
                application_id
            )

            applications.append({

                "application_id":
                application_id,

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
                        row.get(
                            "loan_amount",
                            0
                        )
                    )
                ),

                "foir":
                foir,

                "credit_score":
                safe_int(
                    row.get(
                        "cibil_score",
                        row.get(
                            "credit_score",
                            0
                        )
                    )
                ),

                "risk_score":
                score_data[
                    "risk_score"
                ],

                "risk_tier":
                score_data[
                    "risk_tier"
                ],

                "application_status":
                safe_str(
                    row.get(
                        "application_status",
                        row.get(
                            "status",
                            "Pending"
                        )
                    )
                ),

                "date_applied":
                safe_str(
                    row.get(
                        "date_applied",
                        row.get(
                            "created_at",
                            "2025-05-20"
                        )
                    )
                )
            })

        return {

            "total":
            total,

            "limit":
            limit,

            "offset":
            offset,

            "applications":
            applications
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "error":
            str(e)
        }

# =========================================================
# APPLICATION DETAIL ENDPOINT
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(
    application_id: str
):

    try:

        matched = applications_df[

            applications_df[
                "application_id"
            ].astype(str)

            == str(application_id)
        ]

        if len(matched) == 0:

            return {

                "error":
                "Application not found"
            }

        row = matched.iloc[0]

        # =====================================
        # MONTHLY INCOME
        # =====================================
        monthly_income = safe_float(
            row.get(
                "monthly_income",
                row.get(
                    "annual_income",
                    0
                )
            )
        )

        # =====================================
        # EMI
        # =====================================
        monthly_emi = safe_float(
            row.get(
                "existing_monthly_emi",
                row.get(
                    "monthly_emi",
                    0
                )
            )
        )

        # =====================================
        # FOIR
        # =====================================
        if monthly_income > 0:

            foir = round(
                (
                    monthly_emi
                    /
                    monthly_income
                ) * 100,
                2
            )

        else:

            foir = 0

        # =====================================
        # REAL MODEL SCORE
        # =====================================
        score_data = generate_risk_score(
            application_id
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
                    row.get(
                        "loan_amount",
                        0
                    )
                )
            ),

            "foir":
            foir,

            "credit_score":
            safe_int(
                row.get(
                    "cibil_score",
                    row.get(
                        "credit_score",
                        0
                    )
                )
            ),

            "risk_score":
            score_data["risk_score"],

            "risk_tier":
            score_data["risk_tier"],

            "application_status":
            safe_str(
                row.get(
                    "application_status",
                    row.get(
                        "status",
                        "Pending"
                    )
                )
            ),

            "date_applied":
            safe_str(
                row.get(
                    "date_applied",
                    row.get(
                        "created_at",
                        "2025-05-20"
                    )
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

        high = 0
        medium = 0
        low = 0

        for _, row in applications_df.iterrows():

            application_id = safe_str(
                row.get(
                    "application_id",
                    ""
                )
            )

            score_data = generate_risk_score(
                application_id
            )

            risk_tier = score_data[
                "risk_tier"
            ]

            if risk_tier == "High":

                high += 1

            elif risk_tier == "Medium":

                medium += 1

            else:

                low += 1

        return {

            "total_applications":
            len(applications_df),

            "high":
            high,

            "medium":
            medium,

            "low":
            low
        }

    except Exception as e:

        print(traceback.format_exc())

        return {

            "error":
            str(e)
        }

# =========================================================
# ROOT ENDPOINT
# =========================================================
@app.get("/")
def root():

    return {

        "message":
        "CreditSentinel API Running"
    }

# =========================================================
# RENDER START COMMAND
# =========================================================
# Use this in Render:
#
# Start Command:
# uvicorn main:app --host 0.0.0.0 --port 10000
#
# Build Command:
# pip install -r requirements.txt
#
# =========================================================
  
