# =========================================================
# CREDITSENTINEL FASTAPI - RENDER READY
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
import os

from typing import List

# =========================================================
# IMPORT FEATURE ENGINE
# =========================================================
from feature_engine import compute_features

# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load(
    os.path.join(
        BASE_DIR,
        "creditsentinel_model_v1.pkl"
    )
)

print("✅ Real model loaded")
print(type(model))

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

print("✅ CSV files loaded")

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Loan Risk Scoring API",
    description="API for predicting loan application risk tiers",
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
# HOME ENDPOINT
# =========================================================
@app.get("/")
def home():

    return {
        "message":
        "CreditSentinel API is running successfully",

        "docs":
        "/docs"
    }

# =========================================================
# HEALTH ENDPOINT
# =========================================================
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "features": 43
    }

# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        # =================================================
        # STEP 1: COMPUTE FEATURES
        # =================================================
        features_dict = compute_features(
            req.application_id
        )

        # =================================================
        # STEP 2: MODEL FEATURES
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

                "model_loaded":
                False,

                "error":
                "Model feature names not found"
            }

        # =================================================
        # STEP 3: FILTER FEATURES
        # =================================================
        features_filtered = {}

        for feature in model_features:

            features_filtered[feature] = (
                features_dict.get(feature, 0)
            )

        # =================================================
        # STEP 4: DATAFRAME
        # =================================================
        features_df = pd.DataFrame(
            [features_filtered]
        )

        features_df = features_df[
            model_features
        ]

        features_df = features_df.fillna(0)

        # =================================================
        # STEP 5: SAFE NUMERIC
        # =================================================
        for col in features_df.columns:

            try:

                features_df[col] = pd.to_numeric(
                    features_df[col]
                )

            except:
                pass

        # =================================================
        # STEP 6: PREDICT
        # =================================================
        prediction = model.predict_proba(
            features_df
        )

        risk_score = prediction[:,1][0]

        risk_score = round(
            float(risk_score),
            4
        )

        # =================================================
        # STEP 7: RISK TIER
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

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            False,

            "error":
            str(e)
        }

# =========================================================
# GET APPLICATIONS
# =========================================================
@app.get("/api/applications")
def get_applications():

    applications = []

    for _, row in applications_df.iterrows():

        application = {

            "application_id":
            str(row.get("application_id", "")),

            "applicant_name":
            str(row.get("applicant_name", "")),

            "monthly_income":
            float(row.get("monthly_income", 0)),

            "loan_amount":
            float(row.get("requested_loan_amount", 0)),

            "credit_score":
            int(row.get("cibil_score", 0)),

            "application_status":
            str(row.get("application_status", "Pending")),

            "date_applied":
            str(row.get("date_applied", ""))
        }

        applications.append(application)

    return {

        "total":
        len(applications),

        "applications":
        applications
    }

# =========================================================
# SINGLE APPLICATION
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

    matched = applications_df[
        applications_df["application_id"] == application_id
    ]

    if len(matched) == 0:

        return {
            "error":
            "Application not found"
        }

    row = matched.iloc[0]

    monthly_income = float(
        row.get("monthly_income", 0)
    )

    monthly_emi = float(
        row.get("existing_monthly_emi", 0)
    )

    foir = 0

    if monthly_income > 0:

        foir = round(
            (monthly_emi / monthly_income) * 100,
            2
        )

    score_result = score_application(
        ScoreRequest(
            application_id=application_id
        )
    )

    return {

        "application_id":
        str(row.get("application_id", "")),

        "applicant_name":
        str(row.get("applicant_name", "")),

        "monthly_income":
        monthly_income,

        "loan_amount":
        float(row.get("requested_loan_amount", 0)),

        "foir":
        foir,

        "risk_score":
        score_result.get("risk_score", 0),

        "risk_tier":
        score_result.get("risk_tier", "Low"),

        "credit_score":
        int(row.get("cibil_score", 0)),

        "application_status":
        str(row.get("application_status", "Pending")),

        "date_applied":
        str(row.get("date_applied", ""))
    }

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():

    return {

        "total_applications":
        len(applications_df),

        "approved":
        int(
            (applications_df["application_status"] == "Approved").sum()
        ),

        "rejected":
        int(
            (applications_df["application_status"] == "Rejected").sum()
        ),

        "pending":
        int(
            (applications_df["application_status"] == "Pending").sum()
        )
            }
