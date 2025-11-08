# streamlit_churn_app.py
"""
Robust Streamlit app that uses the pipeline's exact expected input columns
to avoid "columns are missing" errors.

Place:
 - models/final_churn_xgb_pipeline.pkl  (required, full pipeline)
 - (optional) models/feature_names.json  (raw input column names saved during export)
 - (optional) models/sample_input.csv    (raw sample rows)

Run:
    streamlit run streamlit_churn_app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List, Any

# ----------------------------
# feature_engineering_fn (must match training if pipeline used it)
# ----------------------------
def feature_engineering_fn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure expected columns exist
    for col in [
        "PaymentMethod", "StreamingTV", "StreamingMovies", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "MonthlyCharges", "TotalCharges", "tenure"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["is_auto_payment"] = df["PaymentMethod"].apply(
        lambda x: 1 if isinstance(x, str) and "automatic" in x.lower() else 0
    )
    df["tenure_per_monthly"] = df["tenure"] / (df["MonthlyCharges"].replace(0, np.nan) + 1e-6)
    df["avg_monthly_from_total"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan) + 1e-6)
    df["has_streaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["has_support_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["OnlineBackup"] == "Yes") |
        (df["DeviceProtection"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)
    return df

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "models/final_churn_xgb_pipeline.pkl"
FEATURES_PATH = "models/feature_names.json"
SAMPLE_PATH = "models/sample_input.csv"
DEFAULT_THRESHOLD = 0.5

st.set_page_config(page_title="Churn Predictor — Exact-column UI", layout="wide")
st.title("📊 Customer Churn Predictor — Exact-column UI")

# ----------------------------
# Load pipeline + optional artifacts
# ----------------------------
@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        p = joblib.load(path)
        return p, None
    except Exception as e:
        return None, f"Failed to load pipeline: {e}"

@st.cache_data
def load_json_list(path: str) -> Optional[List[str]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_sample_df(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

pipe, load_err = load_pipeline(MODEL_PATH)
if load_err:
    st.error(load_err)
    st.stop()

feature_names_json = load_json_list(FEATURES_PATH)
sample_df = load_sample_df(SAMPLE_PATH)

# ----------------------------
# Utility: infer expected columns (try multiple ways)
# ----------------------------
def infer_expected_raw_columns(pipeline, feature_names_json: Optional[List[str]], sample: Optional[pd.DataFrame]) -> List[str]:
    # 1) Use feature_names.json if provided
    if feature_names_json:
        return list(feature_names_json)
    # 2) Use sample header if present
    if sample is not None:
        return list(sample.columns)
    # 3) Try to inspect pipeline for 'feature_names_in_' attributes
    try:
        if hasattr(pipeline, "feature_names_in_"):
            return list(pipeline.feature_names_in_)
        # search named_steps
        if hasattr(pipeline, "named_steps"):
            for step in pipeline.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass
    # 4) Fallback: sensible telco default (raw columns)
    return [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

expected_raw_cols = infer_expected_raw_columns(pipe, feature_names_json, sample_df)

# Normalize helpers
def normalize_name(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

expected_norm_map = {normalize_name(c): c for c in expected_raw_cols}

# ----------------------------
# UI helper defaults
# ----------------------------
DEFAULT_CATEGORIES = {
    "gender": ["Female", "Male"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phoneservice": ["Yes", "No"],
    "multiplelines": ["No", "Yes", "No phone service"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "onlinesecurity": ["Yes", "No", "No internet service"],
    "onlinebackup": ["Yes", "No", "No internet service"],
    "deviceprotection": ["Yes", "No", "No internet service"],
    "techsupport": ["Yes", "No", "No internet service"],
    "streamingtv": ["Yes", "No", "No internet service"],
    "streamingmovies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperlessbilling": ["Yes", "No"],
    "paymentmethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "seniorcitizen": ["0", "1"]
}

def sample_options_for(expected_col: str) -> Optional[List[Any]]:
    if sample_df is None:
        return None
    # attempt to match sample column case-insensitively
    for c in sample_df.columns:
        if normalize_name(c) == normalize_name(expected_col):
            vals = sample_df[c].dropna().unique().tolist()
            if 1 < len(vals) <= 200:
                return list(vals)
    return None

def is_numeric_col(expected_col: str) -> bool:
    if sample_df is not None:
        for c in sample_df.columns:
            if normalize_name(c) == normalize_name(expected_col):
                return pd.api.types.is_numeric_dtype(sample_df[c])
    # fallback by name
    return any(kw in expected_col.lower() for kw in ["tenure", "charge", "charges", "total", "monthly", "num", "count", "age", "senior"])

def coerce_and_build_row(user_inputs: dict) -> pd.DataFrame:
    """
    Build a DataFrame with exact expected_raw_cols as columns (in order),
    mapping values from user_inputs case-insensitively (normalize keys).
    Missing expected columns are filled: numeric -> 0, others -> "".
    """
    # normalize user keys
    user_norm = {normalize_name(k): v for k, v in user_inputs.items()}
    row = {}
    for exp in expected_raw_cols:
        n = normalize_name(exp)
        if n in user_norm:
            row[exp] = user_norm[n]
        else:
            # try fuzzy match by removing non-alnum; already done by normalize_name
            # if still missing: fill numeric-like with 0, else empty string
            if is_numeric_col(exp):
                row[exp] = 0
            else:
                row[exp] = ""
    df_row = pd.DataFrame([row], columns=expected_raw_cols)
    # attempt numeric coercion for numeric-like columns
    for c in df_row.columns:
        if is_numeric_col(c):
            df_row[c] = pd.to_numeric(df_row[c], errors="coerce").fillna(0)
    return df_row

def predict_proba_with_fallback(pipeline, df: pd.DataFrame) -> (np.ndarray, str):
    # try predict_proba
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(df)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return np.asarray(probs)[:, 1], "predict_proba"
            return np.asarray(probs).ravel(), "predict_proba"
        except Exception:
            pass
    # try decision_function
    if hasattr(pipeline, "decision_function"):
        try:
            dec = pipeline.decision_function(df)
            arr = np.asarray(dec).ravel()
            return 1.0 / (1.0 + np.exp(-arr)), "decision_function"
        except Exception:
            pass
    # fallback to predict labels
    if hasattr(pipeline, "predict"):
        preds = pipeline.predict(df)
        return np.asarray([float(p) for p in preds]), "predict"
    raise RuntimeError("Model does not support any prediction interface.")

# ----------------------------
# Build UI: using expected_raw_cols exactly for labels/keys
# ----------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Churn threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
st.sidebar.write(f"Model path: `{MODEL_PATH}`")
st.sidebar.write(f"Expecting {len(expected_raw_cols)} input columns (will be sent in this order).")
st.sidebar.markdown("---")

mode = st.radio("Mode", ["Single Input", "Batch CSV"], index=0, horizontal=True)

if mode == "Single Input":
    st.header("Single customer input (fields match model expectation)")
    st.info("Fields below are the exact columns the model expects; select values where available.")

    # layout 2 columns
    cols = st.columns(2)
    user_inputs = {}
    for i, colname in enumerate(expected_raw_cols):
        c = cols[i % 2]
        # try sample options first
        opts = sample_options_for(colname)
        if opts is not None:
            sel = c.selectbox(colname, [""] + [str(v) for v in opts], index=0, key=f"ui_{i}")
            user_inputs[colname] = "" if sel == "" else sel
            continue
        # next use default category mapping if available
        default_opts = DEFAULT_CATEGORIES.get(colname.lower())
        if default_opts is not None:
            sel = c.selectbox(colname, [""] + default_opts, index=0, key=f"def_{i}")
            user_inputs[colname] = "" if sel == "" else sel
            continue
        # numeric?
        if is_numeric_col(colname):
            val = c.number_input(colname, value=0.0, key=f"num_{i}", format="%.4f")
            user_inputs[colname] = val
            continue
        # fallback free text
        val = c.text_input(colname, value="", key=f"txt_{i}")
        user_inputs[colname] = val

    st.subheader("Preview (this is what you'll submit)")
    st.dataframe(pd.DataFrame([user_inputs]).astype(object))

    if st.button("Predict"):
        try:
            df_to_predict = coerce_and_build_row(user_inputs)
            st.subheader("Aligned input (sent to pipeline):")
            st.dataframe(df_to_predict.astype(object))
            probs, method = predict_proba_with_fallback(pipe, df_to_predict), None
            # predict_proba_with_fallback returns tuple (arr, method) earlier; adjust:
            # We returned (array, method) originally; here fix to accept both styles:
            if isinstance(probs, tuple) and len(probs) == 2:
                probs_arr, used = probs
            else:
                # our function returns (arr, method) so handle normally
                try:
                    probs_arr, used = predict_proba_with_fallback(pipe, df_to_predict)
                except Exception as ee:
                    st.error(f"Prediction error: {ee}")
                    raise

            p = float(np.asarray(probs_arr).ravel()[0])
            label = int(p >= threshold)
            st.success(f"Churn probability: {p:.4f}  (method: {used})")
            st.info(f"Predicted label (churn=1): {label}")

            # download single prediction
            out = df_to_predict.copy()
            out["pred_proba"] = p
            out["pred_label"] = label
            st.download_button("Download prediction CSV", out.to_csv(index=False).encode("utf-8"),
                               file_name="single_prediction.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.header("Batch CSV prediction (columns mapped case-insensitively)")
    st.write("Upload a CSV whose columns roughly match the model's expected raw column names (case-insensitive).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            st.write(f"Loaded {df_batch.shape[0]} rows and {df_batch.shape[1]} columns.")
            st.dataframe(df_batch.head(10))

            if st.button("Run batch prediction"):
                try:
                    aligned_rows = []
                    for _, row in df_batch.iterrows():
                        user_row = row.to_dict()
                        # coerce strings for keys -> mapping function expects keys as entered; we send actual names
                        aligned = coerce_and_build_row(user_row if isinstance(user_row, dict) else dict(user_row))
                        aligned_rows.append(aligned.iloc[0].to_dict())
                    df_aligned = pd.DataFrame(aligned_rows, columns=expected_raw_cols)
                    # ensure numeric columns are numeric
                    for c in df_aligned.columns:
                        if is_numeric_col(c):
                            df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce").fillna(0)

                    probs_arr, used_method = predict_proba_with_fallback(pipe, df_aligned)
                    df_aligned["pred_proba"] = probs_arr
                    df_aligned["pred_label"] = (df_aligned["pred_proba"] >= threshold).astype(int)
                    st.success("Batch predictions complete")
                    st.dataframe(df_aligned.head(20))
                    st.download_button("Download predictions CSV", df_aligned.to_csv(index=False).encode("utf-8"),
                                       file_name="predictions_batch.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

st.write("---")
st.caption("This app constructs an aligned DataFrame with exact expected pipeline column names, so the pipeline will not raise missing-column errors. Save models/feature_names.json or models/sample_input.csv during export to make the UI friendlier.")
