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

loan_df = pd.read_csv(
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

        print("\n================================")
        print("APPLICATION:", req.application_id)
        print("================================")

        # =================================================
        # STEP 2: GET MODEL FEATURES
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

        print(
            "Model expects:",
            len(model_features),
            "features"
        )

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
        # STEP 4: CREATE DATAFRAME
        # =================================================
        features_df = pd.DataFrame(
            [features_filtered]
        )

        # =================================================
        # STEP 5: CORRECT FEATURE ORDER
        # =================================================
        features_df = features_df[
            model_features
        ]

        # =================================================
        # STEP 6: CLEAN VALUES
        # =================================================
        features_df = features_df.fillna(0)

        # =================================================
        # STEP 7: CONVERT NUMERIC
        # =================================================
        for col in features_df.columns:

            try:

                features_df[col] = pd.to_numeric(
                    features_df[col]
                )

            except:

                pass

        # =================================================
        # DEBUG
        # =================================================
        print("\n========== FEATURES ==========")

        for col in features_df.columns[:10]:

            print(
                col,
                "=",
                features_df.iloc[0][col]
            )

        print("==============================")

        # =================================================
        # STEP 8: PREDICT
        # =================================================
        prediction = model.predict_proba(
            features_df
        )

        print("\nPrediction Array:")
        print(prediction)

        risk_score = prediction[:,1][0]

        risk_score = round(
            float(risk_score),
            4
        )

        # =================================================
        # STEP 9: RISK TIER
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
# SCORE BATCH APPLICATIONS
# =========================================================
@app.post("/api/score-batch")
def score_batch(req: BatchScoreRequest):

    results = []

    for app_id in req.application_ids:

        try:

            # =============================================
            # GET FEATURES
            # =============================================
            features_dict = compute_features(
                app_id
            )

            # =============================================
            # GET MODEL FEATURES
            # =============================================
            if hasattr(model, 'feature_names_in_'):

                model_features = list(
                    model.feature_names_in_
                )

            else:

                model_features = list(
                    model.feature_name_
                )

            # =============================================
            # FILTER FEATURES
            # =============================================
            features_filtered = {

                name: features_dict.get(name, 0)

                for name in model_features
            }

            # =============================================
            # DATAFRAME
            # =============================================
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

            # =============================================
            # PREDICT
            # =============================================
            risk_score = model.predict_proba(
                features_df
            )[:,1][0]

            risk_score = round(
                float(risk_score),
                4
            )

            # =============================================
            # RISK TIER
            # =============================================
            if risk_score < 0.3:

                tier = "Low"

            elif risk_score < 0.6:

                tier = "Medium"

            else:

                tier = "High"

            # =============================================
            # APPEND RESULT
            # =============================================
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
@app.get("/api/applications")
def get_applications():
    applications = db.query(Application).all()
    
    return {
        "total": len(applications),
        "applications": [
            {
                "application_id": app.id,
                "applicant_name": app.name,
                "monthly_income": app.monthly_income,      # ← ADD THIS LINE
                "loan_amount": app.loan_amount,
                "foir": (app.monthly_emi / app.monthly_income * 100) if app.monthly_income > 0 else 0,  # ← ADD THIS LINE
                "risk_score": app.risk_score,
                "risk_tier": app.risk_tier,
                "credit_score": app.cibil_score,
                "application_status": app.status,
                "date_applied": app.created_at
            }
            for app in applications
        ]
    }
# =========================================================
# APPLICATION DETAIL
# =========================================================
@app.get("/api/applications/{application_id}")
def get_application_detail(
    application_id: str
):

    red_flags = []

    if application_id == "APP-000004":

        red_flags = [

            "High FOIR",
            "Low Income",
            "Large Loan Amount"
        ]

    return {

        "application_id":
        application_id,

        "applicant_name":
        "Rahul Yadav",

        "monthly_income":
        55107,

        "requested_loan_amount":
        390000,

        "foir":
        26.48,

        "cibil_score":
        706,

        "employment_type":
        "Self-Employed",

        "employment_years":
        1.0,

        "risk_score":
        39.2,

        "risk_tier":
        "Low",

        "red_flags":
        red_flags,

        "memo_available":
        False
    }

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================

@app.get("/api/portfolio/summary")
def portfolio_summary():

    return {

        "total_applications": 15000,
        "approved": 8200,
        "rejected": 4100,
        "pending": 2700
    }

