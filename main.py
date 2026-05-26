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
@app.get("/api/applications")
def get_applications():

    applications = [

        {
            "application_id": "APP-000001",
            "applicant_name": "Rahul Yadav",
            "risk_score": 0.2145,
            "risk_tier": "low",
            "credit_score": 706,
            "loan_amount": 390000,
            "application_status": "approved",
            "date_applied": "2026-05-20"
        },

        {
            "application_id": "APP-000002",
            "applicant_name": "Priya Sharma",
            "risk_score": 0.5812,
            "risk_tier": "medium",
            "credit_score": 682,
            "loan_amount": 150000,
            "application_status": "pending",
            "date_applied": "2026-05-18"
        },

        {
            "application_id": "APP-000003",
            "applicant_name": "Amit Kumar",
            "risk_score": 0.3411,
            "risk_tier": "low",
            "credit_score": 721,
            "loan_amount": 170000,
            "application_status": "approved",
            "date_applied": "2026-05-15"
        },

        {
            "application_id": "APP-000004",
            "applicant_name": "Sneha Reddy",
            "risk_score": 0.8123,
            "risk_tier": "high",
            "credit_score": 590,
            "loan_amount": 500000,
            "application_status": "rejected",
            "date_applied": "2026-05-12"
        },

        {
            "application_id": "APP-000005",
            "applicant_name": "Vikram Singh",
            "risk_score": 0.6534,
            "risk_tier": "high",
            "credit_score": 745,
            "loan_amount": 800000,
            "application_status": "pending",
            "date_applied": "2026-05-10"
        }
    ]

    return {
        "total": len(applications),
        "applications": applications
    }
# =========================================================
# APPLICATION DETAIL
# =========================================================
# =========================================================
# SINGLE APPLICATION DETAIL ENDPOINT
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(application_id: str):

    applications = {

        "APP-000001": {
            "application_id": "APP-000001",
            "applicant_name": "Rahul Yadav",
            "risk_score": 0.2145,
            "risk_tier": "low",
            "credit_score": 706,
            "loan_amount": 390000,
            "application_status": "approved",
            "date_applied": "2026-05-20"
        },

        "APP-000002": {
            "application_id": "APP-000002",
            "applicant_name": "Priya Sharma",
            "risk_score": 0.5812,
            "risk_tier": "medium",
            "credit_score": 682,
            "loan_amount": 150000,
            "application_status": "pending",
            "date_applied": "2026-05-18"
        },

        "APP-000003": {
            "application_id": "APP-000003",
            "applicant_name": "Amit Kumar",
            "risk_score": 0.3411,
            "risk_tier": "low",
            "credit_score": 721,
            "loan_amount": 170000,
            "application_status": "approved",
            "date_applied": "2026-05-15"
        },

        "APP-000004": {
            "application_id": "APP-000004",
            "applicant_name": "Sneha Reddy",
            "risk_score": 0.8123,
            "risk_tier": "high",
            "credit_score": 590,
            "loan_amount": 500000,
            "application_status": "rejected",
            "date_applied": "2026-05-12"
        },

        "APP-000005": {
            "application_id": "APP-000005",
            "applicant_name": "Vikram Singh",
            "risk_score": 0.6534,
            "risk_tier": "high",
            "credit_score": 745,
            "loan_amount": 800000,
            "application_status": "pending",
            "date_applied": "2026-05-10"
        }
    }

    # ==========================================
    # RETURN MATCHING APPLICATION
    # ==========================================
    if application_id in applications:

        return applications[application_id]

    return {
        "error": "Application not found"
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
