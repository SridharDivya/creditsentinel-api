import datetime
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List
import os

class Config:

    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "creditsentinel_model_v1.pkl"
    )
# LOAD TRAINED MODEL
from config import Config
model = joblib.load(Config.MODEL_PATH)

print("✅ Real model loaded")
print(type(model))
features_dict = compute_features(
    "APP-000001"
)

print(type(features_dict))
print(len(features_dict))
print(features_dict)
# =========================================================
# CreditSentinel ML API - FINAL FIXED VERSION
# DIFFERENT RISK SCORES FOR DIFFERENT APPLICATIONS
# Google Colab Compatible
# =========================================================

# =========================================================


# =========================================================
# IMPORT FEATURE ENGINE
# =========================================================
from feature_engine import compute_features

# =========================================================
# FIX COLAB EVENT LOOP ISSUE
# =========================================================
nest_asyncio.apply()

# =========================================================
# LOAD MODEL
# =========================================================
from config import Config
model = joblib.load(Config.MODEL_PATH)

print("✅ Real model loaded")
print(type(model))
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
# SCORE ENDPOINT
# =========================================================
@app.post("/api/score")
def score_application(req: ScoreRequest):

    try:

        # =================================================
        # STEP 1: FETCH FEATURES
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
                "application_id": req.application_id,
                "model_loaded": False,
                "error": "Model feature names not found"
            }

        # =================================================
        # STEP 3: FILTER FEATURES
        # =================================================
        features_filtered = {}

        for feature in model_features:

            if feature in features_dict:

                features_filtered[feature] = (
                    features_dict[feature]
                )

            else:

                features_filtered[feature] = 0

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
        # STEP 5: CONVERT NUMERIC
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

        risk_score = prediction[:, 1][0]

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

            if hasattr(model, 'feature_names_in_'):

                model_features = list(
                    model.feature_names_in_
                )

            else:

                model_features = list(
                    model.feature_name_
                )

            features_filtered = {
                name: features_dict.get(name, 0)
                for name in model_features
            }

            features_df = pd.DataFrame(
                [features_filtered]
            )

            features_df = features_df[
                model_features
            ]

            features_df = features_df.fillna(0)

            # =============================================
            # SAFE NUMERIC CONVERSION
            # =============================================
            for col in features_df.columns:

                try:

                    features_df[col] = pd.to_numeric(
                        features_df[col]
                    )

                except:

                    pass

            risk_score = model.predict_proba(
                features_df
            )[:,1][0]

            risk_score = round(
                float(risk_score),
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
                tier,

                "features_used":
                len(features_filtered)
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
# APPLICATIONS ENDPOINT
# =========================================================
# =========================================================
# APPLICATION LIST ENDPOINT
# =========================================================
# =========================================================
# APPLICATION LIST ENDPOINT
# =========================================================
# =========================================================
# APPLICATION LIST ENDPOINT
# =========================================================
# =========================================================
# APPLICATION LIST ENDPOINT
# =========================================================
@app.get("/api/applications")
def get_applications():

    applications = []

    for _, row in applications_df.iterrows():

        application_id = str(
            row.get("application_id", "")
        )

        applicant_name = str(
            row.get("applicant_name", "")
        )

        monthly_income = float(
            row.get("monthly_income", 0)
        )

        loan_amount = float(
            row.get(
                "requested_loan_amount",
                0
            )
        )

        existing_monthly_emi = float(
            row.get(
                "existing_monthly_emi",
                0
            )
        )

        # =========================================
        # CALCULATE FOIR
        # =========================================
        if monthly_income > 0:

            foir = round(
                (
                    existing_monthly_emi
                    / monthly_income
                ) * 100,
                2
            )

        else:

            foir = 0

        # =========================================
        # GET LIVE RISK SCORE
        # =========================================
        score_result = score_application(
            ScoreRequest(
                application_id=application_id
            )
        )

        risk_score = score_result.get(
            "risk_score",
            0
        )

        risk_tier = score_result.get(
            "risk_tier",
            "Low"
        )

        # =========================================
        # FINAL RESPONSE RECORD
        # =========================================
        applications.append({

            "application_id":
            application_id,

            "applicant_name":
            applicant_name,

            # IMPORTANT FIXES
            "monthly_income":
            monthly_income,

            "foir":
            foir,

            "loan_amount":
            loan_amount,

            "risk_score":
            risk_score,

            "risk_tier":
            risk_tier,

            "credit_score":
            int(
                row.get(
                    "cibil_score",
                    0
                )
            ),

            "application_status":
            str(
                row.get(
                    "application_status",
                    "Pending"
                )
            ),

            "date_applied":
            str(
                row.get(
                    "date_applied",
                    ""
                )
            )
        })

    return {

        "total":
        len(applications),

        "applications":
        applications
    }
# =========================================================
# APPLICATION DETAIL
# =========================================================
# =========================================================
# SINGLE APPLICATION DETAIL ENDPOINT
# =========================================================
# =========================================================
# SINGLE APPLICATION DETAIL ENDPOINT
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

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

    # =========================================
    # GET MONTHLY INCOME
    # =========================================
    monthly_income = float(
        row.get("monthly_income", 0)
    )

    # =========================================
    # GET MONTHLY EMI
    # =========================================
    monthly_emi = float(
        row.get(
            "existing_monthly_emi",
            0
        )
    )

    # =========================================
    # CALCULATE FOIR
    # =========================================
    if monthly_income > 0:

        foir = round(
            (
                monthly_emi
                / monthly_income
            ) * 100,
            2
        )

    else:

        foir = 0

    # =========================================
    # GET LIVE SCORE
    # =========================================
    score_result = score_application(
        ScoreRequest(
            application_id=application_id
        )
    )

    return {

        "application_id":
        str(
            row.get(
                "application_id",
                ""
            )
        ),

        "applicant_name":
        str(
            row.get(
                "applicant_name",
                ""
            )
        ),

        # IMPORTANT FIXES
        "monthly_income":
        monthly_income,

        "loan_amount":
        float(
            row.get(
                "requested_loan_amount",
                0
            )
        ),

        "foir":
        foir,

        "risk_score":
        score_result.get(
            "risk_score",
            0
        ),

        "risk_tier":
        score_result.get(
            "risk_tier",
            "Low"
        ),

        "credit_score":
        int(
            row.get(
                "cibil_score",
                0
            )
        ),

        "application_status":
        str(
            row.get(
                "application_status",
                "Pending"
            )
        ),

        "date_applied":
        str(
            row.get(
                "date_applied",
                ""
            )
        )
    }
# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
# =========================================================
# PORTFOLIO SUMMARY ENDPOINT
# =========================================================
# =========================================================
# PORTFOLIO SUMMARY ENDPOINT
# REAL MODEL-BASED RISK DISTRIBUTION
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():

    try:

        # ============================================
        # ALL APPLICATION IDS
        # ============================================
        application_ids = [

            "APP-000001",
            "APP-000002",
            "APP-000003",
            "APP-000004",
            "APP-000005"
        ]

        # ============================================
        # COUNTERS
        # ============================================
        high = 0
        medium = 0
        low = 0

        results = []

        # ============================================
        # LOOP THROUGH APPLICATIONS
        # ============================================
        for app_id in application_ids:

            try:

                # ====================================
                # FETCH FEATURES
                # ====================================
                features_dict = compute_features(
                    app_id
                )

                # ====================================
                # GET MODEL FEATURE ORDER
                # ====================================
                if hasattr(model, 'feature_names_in_'):

                    model_features = list(
                        model.feature_names_in_
                    )

                else:

                    model_features = list(
                        model.feature_name_
                    )

                # ====================================
                # FILTER FEATURES
                # ====================================
                features_filtered = {

                    name: features_dict.get(name, 0)

                    for name in model_features
                }

                # ====================================
                # CREATE DATAFRAME
                # ====================================
                features_df = pd.DataFrame(
                    [features_filtered]
                )

                features_df = features_df[
                    model_features
                ]

                features_df = features_df.fillna(0)

                features_df = features_df.astype(float)

                # ====================================
                # PREDICT RISK SCORE
                # ====================================
                risk_score = model.predict_proba(
                    features_df
                )[:,1][0]

                risk_score = round(
                    float(risk_score),
                    4
                )

                # ====================================
                # DETERMINE RISK TIER
                # ====================================
                if risk_score < 0.4:

                    tier = "low"
                    low += 1

                elif risk_score < 0.65:

                    tier = "medium"
                    medium += 1

                else:

                    tier = "high"
                    high += 1

                # ====================================
                # STORE RESULT
                # ====================================
                results.append({

                    "application_id":
                    app_id,

                    "risk_score":
                    risk_score,

                    "risk_tier":
                    tier
                })

            except Exception as e:

                results.append({

                    "application_id":
                    app_id,

                    "error":
                    str(e)
                })

        # ============================================
        # FINAL RESPONSE
        # ============================================
        return {

            "total_applications":
            len(application_ids),

            "high":
            high,

            "medium":
            medium,

            "low":
            low,

            "applications":
            results
        }

    except Exception as e:

        return {

            "error":
            str(e)
        }
