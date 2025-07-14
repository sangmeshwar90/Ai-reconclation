from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import numpy as np
import os
import difflib  # ✅ Fuzzy matching
import google.generativeai as genai

# === Gemini Setup ===
genai.configure(api_key="AIzaSyBPbmhB_3Nxnkgn9RrfqfPtgluoqRmKWUM")

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Fuzzy Column Matching ===
def fuzzy_match(colnames, targets):
    matches = []
    for target in targets:
        match = difflib.get_close_matches(target, colnames, n=1, cutoff=0.7)
        if match:
            matches.append(match[0])
    return matches

# === Join Type Picker ===
def pick_join_type(scenario: str) -> str:
    if scenario in ["bank_recon", "generic_recon"]:
        return "outer"
    if scenario in ["loan_payment", "sales_inventory"]:
        return "inner"
    if scenario == "hr_payroll":
        return "left"
    return "outer"

# === Guess Join Keys ===
def guess_join_keys(df1, df2):
    key_patterns = ['id', 'ref', 'code', 'number', 'transaction', 'loan', 'account', 'employee', 'customer', 'invoice', 'order']
    df1_keys = [col for col in df1.columns if any(pat in col.lower() for pat in key_patterns)]
    df2_keys = [col for col in df2.columns if any(pat in col.lower() for pat in key_patterns)]
    common_keys = list(set(df1_keys) & set(df2_keys))
    if common_keys:
        return common_keys[0], common_keys[0]
    if df1_keys and df2_keys:
        return df1_keys[0], df2_keys[0]
    return df1.columns[0], df2.columns[0]

# === Gemini-Powered Scenario Detection ===
def detect_scenario(df1, df2):
    sample1 = df1.head(3).to_csv(index=False)
    sample2 = df2.head(3).to_csv(index=False)

    prompt = f"""
You're a data scientist helping categorize two CSV tables.

You will receive two tables. Based on their content, choose **one** of the following types:
- loan_payment
- bank_recon
- sales_inventory
- hr_payroll
- generic_recon

### Table 1:
{sample1}

### Table 2:
{sample2}

Reply with only the label.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        valid_labels = {"loan_payment", "bank_recon", "sales_inventory", "hr_payroll", "generic_recon"}
        if result in valid_labels:
            return result
    except:
        pass

    return fallback_detect_scenario(df1, df2)

# === Fallback Scenario Detector (non-Gemini) ===
def fallback_detect_scenario(df1, df2):
    all_cols = [col.lower() for col in list(df1.columns) + list(df2.columns)]
    if fuzzy_match(all_cols, ["loanamount", "amountrequested", "loan_amt"]) and fuzzy_match(all_cols, ["amountpaid", "amt_paid", "paid"]):
        return "loan_payment"
    if fuzzy_match(all_cols, ["unitssold", "qtysold"]) and fuzzy_match(all_cols, ["stockremaining", "inventory"]):
        return "sales_inventory"
    if fuzzy_match(all_cols, ["employeeid", "empid"]) and fuzzy_match(all_cols, ["salary", "payroll"]):
        return "hr_payroll"
    if fuzzy_match(all_cols, ["amount", "value", "balance"]):
        return "bank_recon"
    return "generic_recon"

# === Smart Numeric Column Finder ===
def find_best_numeric_col(df, preferred_keywords=None):
    if preferred_keywords is None:
        preferred_keywords = ['amount', 'value', 'loan', 'paid', 'total', 'balance', 'qty', 'stock', 'salary']
    scored_cols = []
    for col in df.columns:
        col_l = col.lower()
        score = sum(1 for kw in preferred_keywords if kw in col_l)
        if pd.api.types.is_numeric_dtype(df[col]) and score > 0 and df[col].nunique() > 1:
            scored_cols.append((col, score))
    if scored_cols:
        return sorted(scored_cols, key=lambda x: -x[1])[0][0]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            return col
    return None

# === Merge with Status Prep ===
def merge_data(df1, df2, key1, key2, scenario):
    how = pick_join_type(scenario)
    merged_df = pd.merge(df1, df2, left_on=key1, right_on=key2, how=how, suffixes=('_left', '_right'))

    col1 = find_best_numeric_col(df1)
    col2 = find_best_numeric_col(df2)

    col1_merged = f"{col1}_left" if f"{col1}_left" in merged_df.columns else col1
    col2_merged = f"{col2}_right" if f"{col2}_right" in merged_df.columns else col2

    merged_df[col1_merged] = pd.to_numeric(merged_df[col1_merged], errors="coerce")
    merged_df[col2_merged] = pd.to_numeric(merged_df[col2_merged], errors="coerce")

    return merged_df, col1_merged, col2_merged, how

# === Scenario-Based Status Logic ===
def apply_status_logic(df: pd.DataFrame, scenario: str, col1: str, col2: str):
    def safe(row, col):
        return row[col] if col in row else np.nan

    def get_status(row):
        val1 = safe(row, col1)
        val2 = safe(row, col2)

        if pd.isna(val1) and pd.notna(val2):
            return "Missing in df1"
        if pd.notna(val1) and pd.isna(val2):
            return "Missing in df2"
        if pd.isna(val1) and pd.isna(val2):
            return "No Data"

        if scenario == "loan_payment":
            if val2 == 0 or pd.isna(val2):
                return "No Payment"
            if np.isclose(val1, val2):
                return "Fully Paid"
            elif val2 < val1:
                return "Partially Paid"
            else:
                return "Overpaid"

        elif scenario == "sales_inventory":
            if val2 <= 0 or pd.isna(val2):
                return "Out of Stock"
            elif val2 < val1:
                return "Low Stock"
            else:
                return "In Stock"

        elif scenario == "hr_payroll":
            return "Paid" if not pd.isna(val2) else "Missing from Payroll"

        if np.isclose(val1, val2, atol=1e-6):
            return "Matched"
        return "Mismatch"

    df["Status"] = df.apply(get_status, axis=1)
    return df

# === Upload Endpoint ===
@app.post("/merge-and-status/")
async def merge_and_status(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        if file1.content_type != "text/csv" or file2.content_type != "text/csv":
            return JSONResponse(status_code=400, content={"error": "Only CSV files are supported"})

        df1 = pd.read_csv(StringIO((await file1.read()).decode()))
        df2 = pd.read_csv(StringIO((await file2.read()).decode()))

        if df1.columns.str.match(r'^\d+$').all() or df2.columns.str.match(r'^\d+$').all():
            return JSONResponse(status_code=400, content={"error": "Missing or invalid CSV headers."})

        key1, key2 = guess_join_keys(df1, df2)
        scenario = detect_scenario(df1, df2)

        merged_df, col1, col2, join_type = merge_data(df1, df2, key1, key2, scenario)
        final_df = apply_status_logic(merged_df, scenario, col1, col2)

        safe_df = final_df.replace({np.nan: None})

        return {
            "message": f"Merged using scenario: {scenario}",
            "join_keys": {"df1": key1, "df2": key2},
            "value_columns": {"df1": col1, "df2": col2},
            "join_type": join_type,
            "columns": list(safe_df.columns),
            "sample": safe_df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing error: {str(e)}"})

# === Download Endpoint ===
@app.get("/download-merged/")
def download_merged(path: str):
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv", filename="merged_result.csv")
    return JSONResponse(status_code=404, content={"error": "File not found"})
