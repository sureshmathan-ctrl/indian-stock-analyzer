# 📊 Indian Stock Analyzer — 150+ Factor Framework

A comprehensive stock analysis tool for Indian equities, covering macro, micro, and sector-specific factors across 150+ data points.

## 🚀 Live Demo
Deploy your own instance for free on Streamlit Community Cloud (see below).

---

## 🌐 Deploy for Free (Streamlit Community Cloud)

### Step 1 — Create a GitHub Repository
1. Go to [github.com](https://github.com) → Sign up / Log in (free)
2. Click **"New repository"**
3. Name it: `indian-stock-analyzer`
4. Set to **Public** *(required for free tier)*
5. Click **"Create repository"**

### Step 2 — Upload These Files
Upload all files from this folder to your new GitHub repo:
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`

*(You can drag & drop files on GitHub's web UI)*

### Step 3 — Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
2. Click **"New app"**
3. Select your repository: `indian-stock-analyzer`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **"Deploy!"**

✅ Your app will be live at: `https://yourname-indian-stock-analyzer.streamlit.app`

---

## 🛠 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Features
- **150+ factor analysis** across macro, micro, sector-specific dimensions
- **Auto-fetches** data via Yahoo Finance (yfinance)
- **Technical indicators** via `ta` library (RSI, MACD, SMAs)
- **Sector-specific** adjustments for 15 Indian sectors
- **Excel export** with color-coded scores and data sources
- **Buy/Sell/Hold** recommendation with composite scoring

## ⚠️ Disclaimer
For educational and research purposes only. Not financial advice. Consult a SEBI-registered advisor before investing.
