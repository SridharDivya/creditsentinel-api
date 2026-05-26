# =========================================================
# CreditSentinel ML API
# RENDER DEPLOYMENT VERSION
# =========================================================

# =========================================================
# IMPORTS
# =========================================================
import os
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from feature_engine import compute_features
from config import Config

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load(
    Config.MODEL_PATH
)

print("✅ Model Loaded")

# =========================================================
# LOAD APPLICATIONS DATA
# =========================================================
applications_df = pd.read_csv(
    Config.APPLICATIONS_PATH
)

print("✅ Applications Data Loaded")
print(applications_df.shape)

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
# HEALTH CHECK
# =========================================================
@app.get("/")
def root():

    return {

        "message":
        "CreditSentinel API Running"
    }


@app.get("/health")
def health():

    return {

        "status":
        "ok",

        "model_loaded":
        True,

        "total_applications":
        len(applications_df)
    }

# =========================================================
# HELPER FUNCTION
# =========================================================
def predict_risk(application_id):

    # =============================================
    # FETCH FEATURES
    # =============================================
    features_dict = compute_features(
        application_id
    )

    # =============================================
    # MODEL FEATURES
    # =============================================
    if hasattr(model, "feature_names_in_"):

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

        feature:
        features_dict.get(feature, 0)

        for feature in model_features
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
    if risk_score < 0.35:

        risk_tier = "Low"

    elif risk_score < 0.70:

        risk_tier = "Medium"

    else:

        risk_tier = "High"

    return risk_score, risk_tier

# =========================================================
# SCORE SINGLE APPLICATION
# =========================================================
@app.post("/api/score")
def score_application(
    req: ScoreRequest
):

    try:

        risk_score, risk_tier = predict_risk(
            req.application_id
        )

        return {

            "application_id":
            req.application_id,

            "model_loaded":
            True,

            "risk_score":
            risk_score,

            "risk_tier":
            risk_tier
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
def score_batch(
    req: BatchScoreRequest
):

    results = []

    high = 0
    medium = 0
    low = 0

    for app_id in req.application_ids:

        try:

            risk_score, risk_tier = predict_risk(
                app_id
            )

            if risk_tier == "High":

                high += 1

            elif risk_tier == "Medium":

                medium += 1

            else:

                low += 1

            results.append({

                "application_id":
                app_id,

                "risk_score":
                risk_score,

                "risk_tier":
                risk_tier
            })

        except Exception as e:

            results.append({

                "application_id":
                app_id,

                "error":
                str(e)
            })

    return {

        "total_applications":
        len(req.application_ids),

        "high":
        high,

        "medium":
        medium,

        "low":
        low,

        "results":
        results
    }

# =========================================================
# APPLICATION LIST
# =========================================================
@app.get("/api/applications")
def get_applications():

    applications = []

    for _, row in applications_df.iterrows():

        try:

            application_id = str(
                row.get(
                    "application_id",
                    ""
                )
            )

            # =====================================
            # LIVE RISK SCORE
            # =====================================
            try:

                risk_score, risk_tier = predict_risk(
                    application_id
                )

            except:

                risk_score = 0.50
                risk_tier = "Medium"

            applications.append({

                "application_id":
                application_id,

                "applicant_name":
                str(row.get(
                    "applicant_name",
                    "Unknown"
                )),

                "monthly_income":
                float(row.get(
                    "monthly_income",
                    row.get("income", 0)
                )),

                "requested_loan_amount":
                float(row.get(
                    "requested_loan_amount",
                    row.get("loan_amount", 0)
                )),

                "foir":
                float(row.get(
                    "foir",
                    row.get("FOIR", 0)
                )),

                "cibil_score":
                int(row.get(
                    "cibil_score",
                    0
                )),

                "risk_score":
                risk_score,

                "risk_tier":
                risk_tier,

                "application_status":
                str(row.get(
                    "application_status",
                    "pending"
                )),

                "date_applied":
                str(row.get(
                    "date_applied",
                    ""
                ))
            })

        except Exception as e:

            print(e)

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
def get_application_detail(
    application_id: str
):

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

    try:

        risk_score, risk_tier = predict_risk(
            application_id
        )

    except:

        risk_score = 0.50
        risk_tier = "Medium"

    return {

        "application_id":
        str(row.get(
            "application_id",
            ""
        )),

        "applicant_name":
        str(row.get(
            "applicant_name",
            ""
        )),

        "monthly_income":
        float(row.get(
            "monthly_income",
            row.get("income", 0)
        )),

        "requested_loan_amount":
        float(row.get(
            "requested_loan_amount",
            row.get("loan_amount", 0)
        )),

        "foir":
        float(row.get(
            "foir",
            row.get("FOIR", 0)
        )),

        "cibil_score":
        int(row.get(
            "cibil_score",
            0
        )),

        "risk_score":
        risk_score,

        "risk_tier":
        risk_tier,

        "application_status":
        str(row.get(
            "application_status",
            "pending"
        )),

        "date_applied":
        str(row.get(
            "date_applied",
            ""
        ))
    }

# =========================================================
# PORTFOLIO SUMMARY
# =========================================================
@app.get("/api/portfolio/summary")
def portfolio_summary():

    high = 0
    medium = 0
    low = 0

    total_exposure = 0

    results = []

    application_ids = list(
        applications_df[
            "application_id"
        ].unique()
    )

    for app_id in application_ids:

        try:

            risk_score, risk_tier = predict_risk(
                app_id
            )

            if risk_tier == "High":

                high += 1

            elif risk_tier == "Medium":

                medium += 1

            else:

                low += 1

            row = applications_df[
                applications_df[
                    "application_id"
                ] == app_id
            ].iloc[0]

            loan_amount = float(
                row.get(
                    "requested_loan_amount",
                    row.get(
                        "loan_amount",
                        0
                    )
                )
            )

            total_exposure += loan_amount

            results.append({

                "application_id":
                app_id,

                "risk_score":
                risk_score,

                "risk_tier":
                risk_tier,

                "loan_amount":
                loan_amount
            })

        except Exception as e:

            results.append({

                "application_id":
                app_id,

                "error":
                str(e)
            })

    return {

        "total_applications":
        len(application_ids),

        "high":
        high,

        "medium":
        medium,

        "low":
        low,

        "total_portfolio_exposure":
        total_exposure,

        "applications":
        results
    }

# =========================================================
# RENDER ENTRY POINT
# =========================================================
if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
