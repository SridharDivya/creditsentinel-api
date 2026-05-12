from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime

app = FastAPI(title="CreditSentinel ML API")

# =====================================================
# ENABLE CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# INPUT SCHEMA
# =====================================================

class ApplicationFeatures(BaseModel):
    application_id: str
    monthly_income: float
    requested_loan_amount: float
    existing_monthly_emi: float
    cibil_score: int
    employment_years: float
    foir: float
    loan_to_income_ratio: float
    is_night_application: int

# =====================================================
# OUTPUT SCHEMA
# =====================================================

class ScoreResponse(BaseModel):
    application_id: str
    risk_score: float
    risk_tier: str
    message: str

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():
    return {
        "message": "CreditSentinel ML API Running"
    }

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "CreditSentinel ML API",
        "date": str(datetime.date.today())
    }

# =====================================================
# APPLICATIONS
# =====================================================

@app.get("/api/applications")
def get_applications():

    return {
        "total": 2,
        "applications": [
            {
                "application_id": "APP-000001",
                "applicant_name": "Rahul Yadav",
                "risk_tier": "Low"
            },
            {
                "application_id": "APP-000002",
                "applicant_name": "Priya Sharma",
                "risk_tier": "Medium"
            }
        ]
    }

# =====================================================
# SCORE ENDPOINT
# =====================================================

@app.post("/api/score", response_model=ScoreResponse)
def score_application(features: ApplicationFeatures):

    risk_score = round(features.foir * 1.2, 1)

    if risk_score > 70:
        tier = "High"
    elif risk_score > 45:
        tier = "Medium"
    else:
        tier = "Low"

    return ScoreResponse(
        application_id=features.application_id,
        risk_score=min(risk_score, 100),
        risk_tier=tier,
        message="Mock response — model not yet connected"
    )