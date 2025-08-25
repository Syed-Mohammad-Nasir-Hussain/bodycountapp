# 🍎 Health & Nutrition Planner

A comprehensive Streamlit app for tracking health, analyzing lab markers, estimating weight loss timelines, calculating macros, and generating AI-powered Indian meal plans.

---

## 🔧 Features

- 🧮 BMI, BMR (Mifflin-St Jeor), TDEE, Body Fat % (US Navy method)
- 🍽️ Macronutrient targets based on your goal
- 🩺 Lab value analysis (Vitamin D, B12, Calcium, HbA1c, Cholesterol, etc.)
- 📉 Weight-loss timeline projection with visual charting
- 🥘 AI meal generator (Indian-focused) using Gemini (Google Generative AI)
- 📊 Macro and weight projection charts
- 📅 Upload weight logs (CSV) to generate 30-day forecasts via Linear Regression
- 🧠 Nutrition analysis of your meals (via Gemini)
- 📝 Downloadable 3-day meal plan with instructions, nutrition, and ingredients

---

## 🚀 Live Demo

👉 [Click here to try the app!](https://nutrition-planner.streamlit.app/) 👈  
Deployed on **Streamlit Cloud**.


---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Syed-Mohammad-Nasir-Hussain/bodycountapp.git
cd bodycountapp
````

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
# .env
API_KEY=your_gemini_api_key_here
```

You can get the API key from Google’s [Generative AI Console](https://aistudio.google.com/app/apikey).

---

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 🧪 Sample Input CSV for Trends

To use the **Weight Trends** feature, upload a CSV with these columns:

```csv
date,weight_kg
2025-08-01,72.5
2025-08-05,72.1
2025-08-10,71.6
```

---

## ☁️ Deploying on Streamlit Cloud

1. Push your project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click “New App” → Select your repo and branch
4. Fill in:

   * **Main file**: `app.py`
   * **Secrets file**: use Streamlit's `Secrets` feature to securely add `API_KEY`
5. Click **Deploy**

---

## 🔐 Environment Variables

* `API_KEY`: Your Gemini (Google Generative AI) API key
  Add it in `.env` for local or via Streamlit Cloud Secrets.

---

## 🧠 Tech Stack

* `Streamlit`: UI framework
* `scikit-learn`: Linear regression for trend forecasting
* `matplotlib`, `seaborn`: Visualization
* `scipy`: Smoothing filters
* `google-generativeai`: Gemini integration (meal ideas, nutrition estimates)
* `dotenv`: Local environment variable management

---

## 📌 Notes

* The app is **not** a substitute for medical advice.
* Body Fat % is estimated using tape measurements (US Navy method).
* Calorie-to-weight change assumes 7700 kcal ≈ 1kg fat (a simplification).
* Meal planning is focused on Indian dietary preferences.

---

## 🙌 Acknowledgments

* Gemini AI (Google) for generative recipes and nutrition analysis
* Streamlit team for an awesome framework


