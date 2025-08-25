# streamlit_app.py
# -------------------------------------------------------------
# Health & Nutrition Planner (Enhanced)
# - Calculates BMI, Body Fat % (US Navy method), BMR, TDEE
# - Macro targets based on goal
# - Accepts recent lab values (Vit D, Calcium, Triglycerides, HDL, LDL, TC, HbA1c, Ferritin, B12)
# - Flags out-of-range values and gives food-focused suggestions
# - Projects weight-loss timeline from chosen calorie deficit
# - Exports a 7-day meal macro template
# - NEW: Charts (matplotlib + seaborn), trend forecasting (scikit-learn),
#        API-based meal ideas via requests (Edamam/USDA), date handling (dateutil),
#        smoothing with scipy
# -------------------------------------------------------------

import os
import math
import io
import json
from datetime import datetime, timedelta
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()  # loads from .env into os.environ


# ---------------------------
# Helpers
# ---------------------------

ACTIVITY_FACTORS = {
    "Sedentary (little/no exercise)": 1.2,
    "Lightly Active (1-3x/week)": 1.375,
    "Moderately Active (3-5x/week)": 1.55,
    "Very Active (6-7x/week)": 1.725,
    "Extra Active (physical job + training)": 1.9,
}

GOALS = {
    "Lose Fat": {"deficit": 0.20},    # 20% deficit default
    "Maintain": {"deficit": 0.00},
    "Gain Muscle": {"deficit": -0.10}, # 10% surplus
}

# Standard-ish micronutrient daily targets (adults). These are simplified reference ranges
# for educational use only (not medical advice). Units noted in tips where relevant.
MICRO_TARGETS = {
    "Vitamin D (25-OH)": {"range": (30, 100), "unit": "ng/mL"},
    "Calcium": {"range": (8.6, 10.2), "unit": "mg/dL"},
    "Triglycerides": {"range": (0, 150), "unit": "mg/dL"},
    "Total Cholesterol": {"range": (0, 200), "unit": "mg/dL"},
    "LDL": {"range": (0, 100), "unit": "mg/dL"},
    "HDL": {"range": (40, 1000), "unit": "mg/dL"},
    "HbA1c": {"range": (4.0, 5.6), "unit": "%"},
    "Ferritin": {"range": (20, 250), "unit": "ng/mL"},
    "Vitamin B12": {"range": (200, 900), "unit": "pg/mL"},
}

FOOD_TIPS = {
    "Vitamin D (25-OH)": "Sunlight (10–20 min/day as tolerated), fortified milk/plant milk, eggs, fatty fish (salmon, mackerel)",
    "Calcium": "Dairy or fortified alternatives, leafy greens, tofu, sesame (tahini)",
    "Triglycerides": "Cut sugar/refined carbs, avoid deep-fried foods, add omega-3 (fish, flax, chia)",
    "Total Cholesterol": "Increase fiber (oats, legumes), reduce saturated fat, choose olive oil, nuts",
    "LDL": "Fiber + plant sterols; prefer lean meats and low-fat dairy",
    "HDL": "Regular exercise, nuts, olive oil, fatty fish",
    "HbA1c": "Lower refined carbs, balanced plate (protein+fiber+healthy fat), regular activity",
    "Ferritin": "Lean red meat, legumes, spinach; pair iron with vitamin C (lemon, citrus)",
    "Vitamin B12": "Animal products, fortified cereals; consider supplements if vegetarian/vegan",
}

# ---------------------------
# Core calculations
# ---------------------------

def bmi(weight_kg: float, height_cm: float) -> float:
    h_m = height_cm / 100
    return weight_kg / (h_m ** 2) if h_m > 0 else np.nan


def bmr_mifflin_st_jeor(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    # Mifflin-St Jeor Equation
    if sex.lower().startswith("m"):  # male
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # female/other default
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def body_fat_us_navy(sex: str, height_cm: float, neck_cm: float, waist_cm: float, hip_cm: float | None = None) -> float:
    """US Navy Method (log10). Requires waist & neck; females also hip."""
    try:
        if sex.lower().startswith("m"):
            # men: BF% = 495/(1.0324 - 0.19077*log10(waist - neck) + 0.15456*log10(height)) - 450
            val = 495 / (
                1.0324 - 0.19077 * math.log10(max(waist_cm - neck_cm, 1e-6)) + 0.15456 * math.log10(height_cm)
            ) - 450
        else:
            # women: BF% = 495/(1.29579 - 0.35004*log10(waist + hip - neck) + 0.22100*log10(height)) - 450
            if hip_cm is None:
                return np.nan
            val = 495 / (
                1.29579 - 0.35004 * math.log10(max(waist_cm + hip_cm - neck_cm, 1e-6)) + 0.22100 * math.log10(height_cm)
            ) - 450
        return float(max(min(val, 75.0), 2.0))  # clamp sensible range
    except Exception:
        return np.nan


def tdee(bmr: float, activity_label: str) -> float:
    factor = ACTIVITY_FACTORS.get(activity_label, 1.2)
    return bmr * factor


def macro_targets(goal: str, weight_kg: float, calories: float) -> dict:
    """
    Protein defaults:
      - Fat loss: 1.8 g/kg
      - Maintain: 1.6 g/kg
      - Gain: 2.0 g/kg
    Fat: 0.8 g/kg minimum (never below 20% calories). Carbs = remaining calories.
    """
    goal_key = goal.lower()
    if "lose" in goal_key:
        p_gkg = 1.8
    elif "gain" in goal_key:
        p_gkg = 2.0
    else:
        p_gkg = 1.6

    protein_g = p_gkg * weight_kg
    fat_g_min = max(0.8 * weight_kg, 0.20 * calories / 9.0)

    calories_after_pf = calories - (protein_g * 4 + fat_g_min * 9)
    carbs_g = max(calories_after_pf / 4.0, 0)

    return {
        "protein_g": round(protein_g, 1),
        "fat_g": round(fat_g_min, 1),
        "carbs_g": round(carbs_g, 1),
    }


def apply_goal_calories(goal: str, tdee_val: float, custom_deficit_pct: float | None = None) -> float:
    if custom_deficit_pct is not None:
        pct = custom_deficit_pct
    else:
        pct = GOALS.get(goal, {"deficit": 0.0})["deficit"]
    return max(tdee_val * (1 - pct), 1200)  # never below 1200 kcal safeguard


def estimate_timeline(current_wt: float, target_wt: float, daily_calorie_diff: float) -> dict:
    """Estimate weight change timeline. Uses 7700 kcal ≈ 1 kg fat."""
    if current_wt <= 0 or target_wt <= 0:
        return {"weeks": np.nan, "days": np.nan, "kcal_per_kg": 7700}

    kg_to_change = target_wt - current_wt
    if daily_calorie_diff == 0:
        return {"weeks": np.inf, "days": np.inf, "kcal_per_kg": 7700}

    total_kcal_needed = abs(kg_to_change) * 7700
    days = total_kcal_needed / abs(daily_calorie_diff)
    return {"weeks": round(days / 7.0, 1), "days": int(days), "kcal_per_kg": 7700}


# ---------------------------
# Gemini-based Recipe Generator
def generate_recipe_with_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API error: {e}"

# Gemini-based Nutrition Breakdown
def analyze_nutrition_with_gemini(food_description: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
You are a nutrition expert. Give me an estimated nutrition breakdown for the following Indian meal:
**{food_description}**

Include:
- Calories
- Protein (g)
- Carbs (g)
- Fats (g)
- Any useful comments or suggestions

Format the answer neatly in markdown.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini API error: {e}"



# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Health & Nutrition Planner", page_icon="🍎", layout="wide")
st.title("🍎 Health & Nutrition Planner")
st.caption("Educational tool. Not medical advice. Consult your clinician for personalized guidance.")


with st.sidebar:
    st.subheader("Your Basics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=0.1)
    with col2:
        weight_kg = st.number_input("Current Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.1)
        target_weight_kg = st.number_input("Target Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)

    st.markdown("---")
    st.subheader("Body Measurements (for Body Fat %)")
    m1, m2, m3 = st.columns(3)
    with m1:
        neck_cm = st.number_input("Neck (cm)", min_value=20.0, max_value=60.0, value=38.0, step=0.1)
    with m2:
        waist_cm = st.number_input("Waist (cm)", min_value=40.0, max_value=200.0, value=90.0, step=0.1)
    with m3:
        hip_cm = st.number_input("Hip (cm, females)", min_value=60.0, max_value=200.0, value=100.0, step=0.1, disabled=sex=="Male")
        hip_val = None if sex=="Male" else hip_cm

    st.markdown("---")
    st.subheader("Lifestyle & Goal")
    activity = st.selectbox("Activity", list(ACTIVITY_FACTORS.keys()), index=1)
    goal = st.selectbox("Goal", list(GOALS.keys()), index=0)
    custom_def = st.slider("Custom calorie deficit/surplus (%)", min_value=-30, max_value=40, value=20 if goal=="Lose Fat" else (0 if goal=="Maintain" else -10))
    custom_def = custom_def / 100.0

    st.markdown("---")
    st.subheader("Optional Lab Values")
    labs = {}
    for k, meta in MICRO_TARGETS.items():
        labs[k] = st.text_input(f"{k} ({meta['unit']})", value="")

# Calculations
bmi_val = round(bmi(weight_kg, height_cm), 1)
bodyfat = body_fat_us_navy(sex, height_cm, neck_cm, waist_cm, hip_val)
bmr_val = round(bmr_mifflin_st_jeor(sex, weight_kg, height_cm, age))
tdee_val = round(tdee(bmr_val, activity))
calories_goal = round(apply_goal_calories(goal, tdee_val, custom_def))
macros = macro_targets(goal, weight_kg, calories_goal)

calorie_diff = tdee_val - calories_goal  # positive = deficit
timeline = estimate_timeline(weight_kg, target_weight_kg, calorie_diff)

# Main layout
summary_tab, macros_tab, labs_tab, plan_tab, charts_tab, meals_tab, trends_tab = st.tabs([
    "Summary", "Macros", "Labs & Tips", "Weekly Plan", "Charts", "Smart Meals (API)", "Weight Trends"
])

with summary_tab:
    st.subheader("Your Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BMI", f"{bmi_val}")
    c2.metric("Body Fat % (est)", f"{(np.nan if math.isnan(bodyfat) else round(bodyfat,1))}")
    c3.metric("BMR", f"{bmr_val} kcal/day")
    c4.metric("TDEE", f"{tdee_val} kcal/day")
    c5.metric("Goal Calories", f"{calories_goal} kcal/day")

    st.markdown("---")
    st.markdown(
        f"**Calorie diff vs TDEE:** {int(calorie_diff)} kcal/day.  "+
        ("(Deficit)" if calorie_diff>0 else ("(Surplus)" if calorie_diff<0 else "(Neutral)"))
    )

    if not (math.isinf(timeline["weeks"]) or math.isnan(timeline["weeks"])):
        st.info(f"Estimated time to reach {target_weight_kg} kg: ~{timeline['weeks']} weeks (~{timeline['days']} days), using 7700 kcal/kg rule.")
    else:
        st.warning("Set a target weight and calorie deficit/surplus to see a timeline.")

with macros_tab:
    st.subheader("Daily Macro Targets")
    st.write(
        pd.DataFrame({
            "Protein (g)": [macros["protein_g"]],
            "Fat (g)": [macros["fat_g"]],
            "Carbs (g)": [macros["carbs_g"]],
            "Calories": [calories_goal],
        })
    )

    st.caption("Protein ~1.6–2.0 g/kg depending on goal. Fat ≥0.8 g/kg (≥20% calories). Carbs = remaining calories.")

with labs_tab:
    st.subheader("Lab Check & Food Tips")

    def parse_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    report = []
    for marker, meta in MICRO_TARGETS.items():
        val = parse_float(labs.get(marker))
        lo, hi = meta["range"]
        flag = "Not provided"
        tip = None
        if val is not None:
            if val < lo:
                flag = f"Low (target {lo}–{hi} {meta['unit']})"
                tip = FOOD_TIPS.get(marker)
            elif val > hi:
                flag = f"High (target {lo}–{hi} {meta['unit']})"
                tip = FOOD_TIPS.get(marker)
            else:
                flag = "Within target"
                tip = "Maintain current habits"
        report.append({"Marker": marker, "Value": val, "Unit": meta['unit'], "Status": flag, "Food Focus": tip})

    df_report = pd.DataFrame(report)
    st.dataframe(df_report, use_container_width=True)
    st.caption("Ranges are generalized for healthy adults and may vary by lab & clinician guidance.")

with plan_tab:
    st.subheader("7-Day Macro Template")
    # Simple 3-meal template per day based on daily macros (unchanged)
    def build_template(macros: dict, calories: int) -> pd.DataFrame:
        # Split into 3 meals + 1 snack (30/30/30/10% calories)
        splits = {
            "Breakfast": 0.30,
            "Lunch": 0.30,
            "Dinner": 0.30,
            "Snack": 0.10,
        }
        rows = []
        start_date = datetime.today().date()
        for day in range(1, 8):
            date_label = (start_date + timedelta(days=day-1)).strftime("%Y-%m-%d")
            for meal, pct in splits.items():
                rows.append({
                    "Day": f"Day {day}",
                    "Date": date_label,
                    "Meal": meal,
                    "Calories": round(calories * pct),
                    "Protein (g)": round(macros["protein_g"] * pct, 1),
                    "Fat (g)": round(macros["fat_g"] * pct, 1),
                    "Carbs (g)": round(macros["carbs_g"] * pct, 1),
                    "Example ideas": "Protein + veg + whole grain (e.g., paneer/chicken + roti/rice + salad)",
                })
        return pd.DataFrame(rows)

    weekly = build_template(macros, calories_goal)
    st.dataframe(weekly, use_container_width=True, height=400)

    # Download CSV
    csv = weekly.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Weekly Macro Template (CSV)",
        data=csv,
        file_name="weekly_macro_template.csv",
        mime="text/csv",
    )

with charts_tab:
    st.subheader("Charts")
    # Macro bar chart
    macro_df = pd.DataFrame({
        "Macro": ["Protein", "Fat", "Carbs"],
        "Grams": [macros["protein_g"], macros["fat_g"], macros["carbs_g"]],
    })
    fig1, ax1 = plt.subplots()
    sns.barplot(data=macro_df, x="Macro", y="Grams", ax=ax1)
    ax1.set_title("Daily Macro Targets (g)")
    st.pyplot(fig1)

    # Weight projection curve using kcal rule, smoothed by Savitzky–Golay
    days = max(timeline.get("days", 0), 1)
    x = np.arange(days + 1)
    daily_delta_kg = (calorie_diff / 7700.0)  # positive deficit -> negative weight change per day when applied
    proj = weight_kg - (x * daily_delta_kg)
    # Smooth if long enough
    if len(proj) >= 7:
        window = 7 if (len(proj) % 2 == 1) else 7 + 1  # window must be odd
        window = min(window, len(proj) - (1 - len(proj) % 2))
        try:
            proj_smooth = savgol_filter(proj, window_length=max(5, window), polyorder=2)
        except Exception:
            proj_smooth = proj
    else:
        proj_smooth = proj

    fig2, ax2 = plt.subplots()
    ax2.plot(x, proj, label="Projection")
    ax2.plot(x, proj_smooth, linestyle="--", label="Smoothed")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Weight (kg)")
    ax2.set_title("Projected Weight Trajectory")
    ax2.legend()
    st.pyplot(fig2)

with meals_tab:
    st.subheader("💡 Smart Meal Generator (Gemini API)")
    st.caption(
        "Ask anything like: 'high protein dosa for breakfast', "
        "'meal with oats and paneer', etc."
    )

    # User input for recipe generation
    recipe_query = st.text_input(
        "Describe your meal idea or ask for a recipe suggestion",
        value="high protein dosa for breakfast"
    )

    if st.button("Generate Recipe"):
        if recipe_query.strip():
            with st.spinner("Generating recipe..."):
                prompt = (
                    f"You're a knowledgeable nutritionist and chef.\n\n"
                    f"Please suggest an Indian recipe based on this request:\n"
                    f"**{recipe_query}**\n\n"
                    f"Include:\n"
                    "- A title\n"
                    "- Ingredients list with quantities (metric units)\n"
                    "- Step-by-step instructions\n"
                    "- Approximate nutrition breakdown per serving (calories, protein g, carbs g, fats g)\n"
                )
                recipe_text = generate_recipe_with_gemini(prompt)
                st.markdown(recipe_text)
        else:
            st.warning("Please enter a description to generate a recipe.")

    st.markdown("---")
    st.subheader("📊 Nutrition Breakdown (Gemini API)")
    nutrition_input = st.text_area(
        "Enter a meal description (e.g., ‘2 roti and eggs’) for nutrition analysis",
        height=100
    )

    if st.button("Analyze Nutrition"):
        if nutrition_input.strip():
            with st.spinner("Analyzing nutrition..."):
                nutrition_prompt = (
                    "You are a nutrition expert.\n\n"
                    "Please provide an estimated nutrition breakdown for this Indian meal:\n"
                    f"**{nutrition_input}**\n\n"
                    "Provide:\n"
                    "- Calories (kcal)\n"
                    "- Protein (g)\n"
                    "- Carbs (g)\n"
                    "- Fats (g)\n"
                    "Optionally, include a brief comment on balance or suggestions.\n\n"
                    "Present the information in a clear, easy-to-read markdown format."
                )
                nutrition_text = analyze_nutrition_with_gemini(nutrition_prompt)
                st.markdown(nutrition_text)
        else:
            st.warning("Please describe the meal for nutrition analysis.")

with trends_tab:
    st.subheader("Upload Weight Log to Forecast (scikit-learn)")
    st.caption("Upload a CSV with columns: date (YYYY-MM-DD), weight_kg. We'll fit a simple linear regression and project 30 days ahead.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        hist = pd.read_csv(file)
        # Basic cleaning
        if "date" in hist.columns and "weight_kg" in hist.columns:
            hist = hist.dropna(subset=["date", "weight_kg"]).copy()
            hist["date"] = hist["date"].apply(lambda d: date_parser.parse(str(d)).date())
            hist = hist.sort_values("date")
            hist["day_index"] = (hist["date"] - hist["date"].min()).dt.days

            X = hist[["day_index"]].values
            y = hist["weight_kg"].values
            model = LinearRegression()
            model.fit(X, y)

            # Forecast next 30 days
            last_idx = hist["day_index"].max()
            future_idx = np.arange(last_idx + 1, last_idx + 31)
            y_pred = model.predict(future_idx.reshape(-1, 1))
            future_dates = [hist["date"].max() + timedelta(days=i) for i in range(1, 31)]
            forecast = pd.DataFrame({"date": future_dates, "predicted_weight": y_pred})

            st.write("**Trend Summary**")
            st.write(f"Slope: {model.coef_[0]:.3f} kg/day | Intercept: {model.intercept_:.2f} kg")

            fig3, ax3 = plt.subplots()
            ax3.plot(hist["date"], hist["weight_kg"], marker="o", label="History")
            ax3.plot(forecast["date"], forecast["predicted_weight"], label="Forecast (30d)")
            ax3.set_title("Weight History & 30-day Forecast")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Weight (kg)")
            ax3.legend()
            st.pyplot(fig3)

            st.dataframe(forecast, use_container_width=True)
        else:
            st.error("CSV must contain 'date' and 'weight_kg' columns.")

st.markdown("""
---
**Notes**
- BMR by Mifflin–St Jeor; TDEE via activity multipliers.
- Body fat % via US Navy method (estimation; tape measurements affect accuracy).
- Weight timeline uses the 7700 kcal/kg heuristic; actual progress varies.
- Charts built with matplotlib + seaborn; trend fit with scikit-learn LinearRegression.
- Smoothing with scipy.savgol_filter. Date parsing/handling via python-dateutil.
- Recipe ideas use the Edamam API if `EDAMAM_APP_ID` and `EDAMAM_APP_KEY` are set; otherwise a local fallback is shown.
- This tool is informational only and not a substitute for professional medical advice.
""")
