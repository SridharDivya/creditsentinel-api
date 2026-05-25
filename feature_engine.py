import pandas as pd
import numpy as np

# =========================
# LOAD DATASETS
# =========================
# Make sure these CSV files are uploaded in Colab
# loan_applications.csv
# bank_statements.csv
# bureau_data.csv
# gst_filings.csv


df_loans = pd.read_csv("loan_applications.csv")
df_bank = pd.read_csv("bank_statements.csv")
df_bur = pd.read_csv("bureau_data.csv")
df_gst = pd.read_csv("gst_filings.csv")


# =========================
# FEATURE ENGINEERING
# =========================

def compute_features(application_id):

    # -------------------------
    # FILTER RECORDS
    # -------------------------
    loan = df_loans[df_loans["application_id"] == application_id].iloc[0]
    bank = df_bank[df_bank["application_id"] == application_id]
    bur = df_bur[df_bur["application_id"] == application_id].iloc[0]
    gst = df_gst[df_gst["application_id"] == application_id]

    # -------------------------
    # BASE FEATURES
    # -------------------------
    features = {
        "monthly_income": loan["monthly_income"],
        "requested_loan_amount": loan["requested_loan_amount"],
        "existing_monthly_emi": loan["existing_monthly_emi"],
        "employment_years": loan["employment_years"],
        "foir": loan["foir"],
        "loan_to_income_ratio": loan["loan_to_income_ratio"],
        "is_night_application": loan["is_night_application"],
    }

    # =========================
    # LOAN FEATURES
    # =========================

    features["short_employment"] = int(loan["employment_years"] < 1)

    features["high_loan_short_emp"] = int(
        features["short_employment"] == 1
        and loan["loan_to_income_ratio"] > 4
    )

    # Dependents feature
    features["dependents"] = int(loan["dependents"]) \
        if "dependents" in loan.index else 0

    # High dependents risk
    features["high_dependents"] = int(features["dependents"] >= 4)

    # Tier-3 city flag
    if "city_tier" in loan.index:
        features["is_tier3"] = int(loan["city_tier"] == 3)
    else:
        features["is_tier3"] = 0

    # Age group risk
    if "age" in loan.index:
        age = loan["age"]

        if age < 21 or age > 58:
            features["age_group_risk"] = 1
        else:
            features["age_group_risk"] = 0
    else:
        features["age_group_risk"] = 0

    # =========================
    # BUREAU FEATURES
    # =========================

    features["cibil_score"] = bur["cibil_score"]
    features["num_credit_inquiries_30d"] = bur["num_credit_inquiries_30d"]
    features["num_credit_inquiries_90d"] = bur["num_credit_inquiries_90d"]
    features["has_previous_default"] = bur["has_previous_default"]
    features["credit_utilization_pct"] = bur["credit_utilization_pct"]
    features["credit_age_months"] = bur["credit_age_months"]
    features["num_active_loans"] = bur["num_active_loans"]

    # Renamed feature expected by model
    features["num_existing_loans"] = bur["num_active_loans"]

    features["low_cibil"] = int(bur["cibil_score"] < 650)

    features["high_inquiries"] = int(
        bur["num_credit_inquiries_30d"] >= 3
    )

    features["foir_cibil_risk"] = int(
        loan["foir"] > 55 and bur["cibil_score"] < 680
    )

    features["high_utilization"] = int(
        bur["credit_utilization_pct"] > 70
    )

    features["inquiry_velocity"] = float(
        bur["num_credit_inquiries_30d"] /
        max(bur["num_credit_inquiries_90d"], 1)
    )

    # Multiple loans flag
    features["multiple_loans"] = int(
        bur["num_active_loans"] >= 3
    )

    # =========================
    # BANK FEATURES
    # =========================

    features["total_emi_bounces"] = int(
        bank["emi_bounces"].sum()
    )

    features["avg_emi_bounces"] = float(
        bank["emi_bounces"].mean()
    )

    features["avg_min_balance"] = float(
        bank["min_eod_balance"].mean()
    )

    features["avg_credits"] = float(
        bank["total_credits"].mean()
    )

    features["income_bank_mismatch"] = float(
        abs(
            loan["monthly_income"] - features["avg_credits"]
        ) / max(loan["monthly_income"], 1) * 100
    )

    features["has_emi_bounces"] = int(
        features["total_emi_bounces"] > 0
    )

    features["low_balance_flag"] = int(
        features["avg_min_balance"] < 5000
    )

    features["inquiry_bounce_combo"] = int(
        features["high_inquiries"] == 1
        and features["has_emi_bounces"] == 1
    )

    # =========================
    # CHEQUE BOUNCE FEATURES
    # =========================

    if "cheque_bounces" in bank.columns:

        features["total_cheque_bounces"] = int(
            bank["cheque_bounces"].sum()
        )

        features["has_cheque_bounces"] = int(
            features["total_cheque_bounces"] > 0
        )

    else:

        features["total_cheque_bounces"] = 0
        features["has_cheque_bounces"] = 0

    # =========================
    # SALARY FEATURES
    # =========================

    if "salary_credit" in bank.columns:

        salary_months = bank[bank["salary_credit"] > 0]

        features["salary_months"] = int(len(salary_months))

        # Irregular salary flag
        features["irregular_salary"] = int(
            len(salary_months) < max(len(bank) * 0.7, 1)
        )

    else:

        features["salary_months"] = 0
        features["irregular_salary"] = 0

    # =========================
    # GST FEATURES
    # =========================

    missing_gst = gst[gst["filing_status"] == "Missing"]

    features["gst_missing_quarters"] = len(missing_gst)

    features["is_self_employed"] = int(
        loan["employment_type"] == "Self-Employed"
    )

    features["self_emp_gst_risk"] = int(
        features["is_self_employed"] == 1
        and features["gst_missing_quarters"] >= 2
    )

    # =========================
    # ADDITIONAL INTERACTION FEATURES
    # =========================

    features["night_high_foir"] = int(
        loan["is_night_application"] == 1
        and loan["foir"] > 50
    )

    # =========================
    # FINAL FEATURE ALIGNMENT
    # =========================
    # Remove extra features not expected by model

    unwanted_features = [
        "loan_amount_risk_score",
        "total_emi_burden",
        "thin_credit_file",
        "overleveraged",
        "utilization_default_risk",
        "avg_max_balance",
        "balance_volatility",
        "gst_compliance_rate",
        "emi_bounce_rate",
        "credit_debit_ratio",
        "cibil_default_combo",
    ]

    for col in unwanted_features:
        if col in features:
            del features[col]

    return features


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    application_id = "APP-000001"

    result = compute_features(application_id)

    print("=" * 60)
    print("FEATURE ENGINEERING OUTPUT")
    print("=" * 60)

    print(f"Application ID: {application_id}")
    print(f"Total Features: {len(result)}")

    print("\nFeature Names:")
    print(list(result.keys()))

    print("\nFeature Values:")

    for key, value in result.items():
        print(f"{key}: {value}")
