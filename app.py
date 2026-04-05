"""
Comprehensive Indian Stock Analysis Tool
Based on the 150+ Factor Framework (India Focus)
Filename: app.py (required for Streamlit Cloud)
"""
import streamlit as st
# st.set_page_config MUST be the first Streamlit command
st.set_page_config(
page_title="Indian Stock Analyzer - 150+ Factor Framework",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded",
)
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import BytesIO
import json
import warnings
warnings.filterwarnings("ignore")
import ta as ta_lib
TA_AVAILABLE = True
SECTOR_MAP = {
"Banking & Financials": "banking",
"IT & Technology": "it",
"Pharma & Healthcare": "pharma",
"Oil, Gas & Energy": "energy",
"FMCG & Consumer": "fmcg",
"Automobile & Auto Ancillary": "auto",
"Metals & Mining": "metals",
"Real Estate & Construction": "realestate",
"Telecom": "telecom",
"Power & Utilities": "power",
"Chemicals & Specialty Chemicals": "chemicals",
"Cement": "cement",
"Defence & Aerospace": "defence",
"Textiles & Apparel": "textiles",
"Other": "other",
}

SCORE_INTERPRETATION = {
(85, 101): ("STRONG BUY", "🟢"),
(70, 85): ("BUY", "🟡"),
(55, 70): ("HOLD", "🟠"),
(40, 55): ("UNDERPERFORM", "🔴"),
(0, 40): ("SELL / AVOID", "⛔"),
}

def get_yahoo_ticker(stock_name):
    stock_name = stock_name.strip().upper()
    if not stock_name.endswith(".NS") and not stock_name.endswith(".BO"):
        return stock_name + ".NS"
    return stock_name

# ─────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────

class StockDataFetcher:
    def __init__(self, ticker_symbol):
        self.symbol = ticker_symbol
        self.yahoo_ticker = get_yahoo_ticker(ticker_symbol)
        self.stock = None
        self.info = {}
        self.history = pd.DataFrame()
        self.data_sources = {}

def fetch_all(self):
    try:
        self.stock = yf.Ticker(self.yahoo_ticker)
        self.info = self.stock.info or {}
        if not self.info or self.info.get("regularMarketPrice") is None:
            if "currentPrice" not in self.info:
                pass
        self.history = self.stock.history(period="2y")
        self.data_sources["Price History"] = (
            "Yahoo Finance (" + self.yahoo_ticker + ") 2yr daily"
        )
        self.data_sources["Company Info"] = (
            "Yahoo Finance " + self.yahoo_ticker + " info"
        )
        self.data_sources["Technical Indicators"] = (
            "Calculated from price history using ta library"
        )
        return True
    except Exception as e:
        st.error(
            "Error fetching data for "
            + self.symbol + ": " + str(e)
        )
        return False

# ─────────────────────────────────────────────
# MACRO DEFAULTS
# ─────────────────────────────────────────────

def get_macro_defaults():
    return {
        "1. Global Macro (12%)": {
            "weight": 12.0,
            "factors": {
                "1.1 Crude Oil Prices": {
                    "sub_weight": 2.5, "default_score": 5,
                    "description": "Brent/WTI trend, OPEC decisions",
                    "source": "tradingeconomics.com/commodity/crude-oil",
                    "how_to_check": "Below $70=Bullish, $70-90=Neutral, Above $90=Bearish",
                },
                "1.2 US Dollar Index (DXY)": {
                    "sub_weight": 2.0, "default_score": 5,
                    "description": "Dollar strength, USD/INR trend",
                    "source": "investing.com/currencies/us-dollar-index",
                    "how_to_check": "DXY<100=Bullish, 100-105=Neutral, >105=Bearish",
                },
                "1.3 US Fed Policy": {
                    "sub_weight": 1.5, "default_score": 5,
                    "description": "Fed funds rate, QE/QT",
                    "source": "federalreserve.gov | CME FedWatch",
                    "how_to_check": "Cuts=Bullish, Pause=Neutral, Hikes=Bearish",
                },
                "1.4 Global GDP Growth": {
                    "sub_weight": 1.0, "default_score": 6,
                    "description": "IMF/World Bank projections",
                    "source": "imf.org/en/Publications/WEO",
                    "how_to_check": ">3%=Bullish, 2-3%=Neutral, <2%=Bearish",
                },
                "1.5 Global Inflation": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "US CPI/PPI, EU inflation",
                    "source": "bls.gov/cpi/ | Eurostat",
                    "how_to_check": "Declining=Bullish, Sticky=Neutral, Rising=Bearish",
                },
                "1.6 Commodity Prices": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "Gold, copper, aluminum, steel",
                    "source": "moneycontrol.com/commodity/ | LME",
                    "how_to_check": "Stable=Bullish consumers, Rising=Bullish producers",
                },
                "1.7 Global Bond Yields": {
                    "sub_weight": 0.8, "default_score": 5,
                    "description": "US 10Y yield, curve",
                    "source": "cnbc.com/quotes/US10Y | fred.stlouisfed.org",
                    "how_to_check": "Declining=Bullish, Rising sharply=Bearish",
                },
                "1.8 Geopolitical Risks": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "Wars, trade wars, sanctions",
                    "source": "Reuters, Bloomberg",
                    "how_to_check": "Low=Bullish(8), Moderate=Neutral(5), High=Bearish(2)",
                },
                "1.9 FII/FPI Flows": {
                    "sub_weight": 1.2, "default_score": 5,
                    "description": "Net FII buying/selling India",
                    "source": "moneycontrol.com FII/DII page | NSDL FPI Monitor",
                    "how_to_check": "Net buyers>5000cr/mo=Bullish, Sellers=Bearish",
                },
            },
        },
        "2. RBI & Monetary (7%)": {
            "weight": 7.0,
            "factors": {
                "2.1 RBI Repo Rate": {
                    "sub_weight": 1.5, "default_score": 6,
                    "description": "Rate, trajectory, stance",
                    "source": "rbi.org.in Monetary Policy",
                    "how_to_check": "Cuts=Bullish, Neutral=Neutral, Hikes=Bearish",
                },
                "2.2 CRR/SLR": {
                    "sub_weight": 0.8, "default_score": 5,
                    "description": "Liquidity impact",
                    "source": "RBI website",
                    "how_to_check": "CRR cut=Bullish, Stable=Neutral, Hike=Bearish",
                },
                "2.3 Liquidity Mgmt": {
                    "sub_weight": 0.7, "default_score": 5,
                    "description": "OMO, LAF, MSF",
                    "source": "RBI Money Market Operations",
                    "how_to_check": "Surplus=Bullish, Balanced=Neutral, Tight=Bearish",
                },
                "2.4 India CPI/WPI": {
                    "sub_weight": 1.0, "default_score": 6,
                    "description": "Core, food, fuel inflation",
                    "source": "mospi.gov.in | RBI",
                    "how_to_check": "CPI<4%=Bullish, 4-6%=Neutral, >6%=Bearish",
                },
                "2.5 INR Stability": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "RBI intervention, REER",
                    "source": "RBI Reference Rate | FBIL",
                    "how_to_check": "Stable=Bullish, Mild depreciation=Neutral",
                },
                "2.6 Credit Growth": {
                    "sub_weight": 0.5, "default_score": 6,
                    "description": "Bank credit growth",
                    "source": "RBI Sectoral Deployment of Bank Credit",
                    "how_to_check": ">15%=Bullish, 10-15%=Neutral, <10%=Bearish",
                },
                "2.7 RBI Sector Norms": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "NPA norms, NBFC regulations",
                    "source": "RBI Circulars & Master Directions",
                    "how_to_check": "Relaxation=Bullish, Tightening=Bearish",
                },
                "2.8 Money Supply M3": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "Broad money growth",
                    "source": "RBI Weekly Statistical Supplement",
                    "how_to_check": ">10%=Bullish, 8-10%=Neutral, <8%=Bearish",
                },
            },
        },
        "3. Govt & Fiscal (8%)": {
            "weight": 8.0,
            "factors": {
                "3.1 Budget & Fiscal Deficit": {
                    "sub_weight": 1.5, "default_score": 6,
                    "description": "Deficit target, capex allocation",
                    "source": "indiabudget.gov.in | CGA",
                    "how_to_check": "Deficit<5.5% + high capex=Bullish",
                },
                "3.2 Tax Policy": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "Corporate tax, cap gains, STT",
                    "source": "Budget documents | CBDT",
                    "how_to_check": "Cuts=Bullish, Hikes=Bearish",
                },
                "3.3 GST Collections": {
                    "sub_weight": 0.8, "default_score": 6,
                    "description": "Rate changes, collections",
                    "source": "pib.gov.in Monthly GST",
                    "how_to_check": ">1.8L Cr/month=Bullish",
                },
                "3.4 PLI Schemes": {
                    "sub_weight": 1.0, "default_score": 7,
                    "description": "Sector PLI, FAME, semiconductor",
                    "source": "DPIIT PLI details",
                    "how_to_check": "Active PLI for sector=Bullish",
                },
                "3.5 Infra Spending": {
                    "sub_weight": 0.8, "default_score": 7,
                    "description": "NIP, Gati Shakti, capex",
                    "source": "Budget Capex | NHAI, Railways",
                    "how_to_check": "Capex>10L Cr/yr growing=Bullish",
                },
                "3.6 Trade Policy": {
                    "sub_weight": 0.7, "default_score": 5,
                    "description": "Duties, anti-dumping, FTAs",
                    "source": "CBIC | DGTR | Commerce Ministry",
                    "how_to_check": "Favorable structure=Bullish",
                },
                "3.7 Disinvestment": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "PSU stake sales",
                    "source": "DIPAM website",
                    "how_to_check": "Active=Bullish for PSUs",
                },
                "3.8 Political Stability": {
                    "sub_weight": 0.7, "default_score": 7,
                    "description": "Govt stability, elections",
                    "source": "News | Election Commission",
                    "how_to_check": "Stable majority=Bullish(8)",
                },
                "3.9 India GDP": {
                    "sub_weight": 1.0, "default_score": 7,
                    "description": "Real GDP, GVA trends",
                    "source": "mospi.gov.in | CSO | RBI",
                    "how_to_check": ">7%=Bullish, 5-7%=Neutral, <5%=Bearish",
                },
            },
        },
        "4. Regulatory (5%)": {
            "weight": 5.0,
            "factors": {
                "4.1 SEBI Regulations": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "Disclosure, margin rules",
                    "source": "sebi.gov.in",
                    "how_to_check": "Investor-friendly=Bullish",
                },
                "4.2 Sector Regulators": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "TRAI, IRDAI, CERC etc",
                    "source": "Respective regulator websites",
                    "how_to_check": "Favorable=Bullish, Adverse=Bearish",
                },
                "4.3 ESG Regulations": {
                    "sub_weight": 0.8, "default_score": 5,
                    "description": "BRSR, carbon tax",
                    "source": "SEBI BRSR | MoEFCC",
                    "how_to_check": "Company ahead of curve=Bullish",
                },
                "4.4 Labour Reforms": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "New labour codes",
                    "source": "Ministry of Labour",
                    "how_to_check": "Simplified=Bullish for manufacturing",
                },
                "4.5 Land Clearances": {
                    "sub_weight": 0.7, "default_score": 5,
                    "description": "Env/forest clearance",
                    "source": "Parivesh portal | MoEFCC",
                    "how_to_check": "Faster=Bullish for infra",
                },
                "4.6 FDI Policy": {
                    "sub_weight": 0.5, "default_score": 6,
                    "description": "Sector caps, route",
                    "source": "DPIIT FDI Policy | RBI",
                    "how_to_check": "Liberal FDI=Bullish",
                },
                "4.7 Digital Regulations": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "DPDP Act, localization",
                    "source": "MeitY | DPDP Act 2023",
                    "how_to_check": "Clear=Neutral, Restrictive=Bearish for tech",
                },
            },
        },
        "5. Market Sentiment (4%)": {
            "weight": 4.0,
            "factors": {
                "5.1 DII Flows": {
                    "sub_weight": 0.7, "default_score": 6,
                    "description": "MF flows, insurance",
                    "source": "AMFI | Moneycontrol FII/DII",
                    "how_to_check": "Strong SIP + DII buying=Bullish",
                },
                "5.2 Nifty Valuation": {
                    "sub_weight": 0.8, "default_score": 5,
                    "description": "Nifty PE vs historical",
                    "source": "niftyindices.com | NSE",
                    "how_to_check": "PE<20=Bullish, 20-24=Neutral, >24=Expensive",
                },
                "5.3 India VIX": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "Fear gauge",
                    "source": "NSE India VIX",
                    "how_to_check": "<13=Bullish, 13-20=Neutral, >20=High fear",
                },
                "5.4 Sector Rotation": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "Sector inflows/outflows",
                    "source": "NSE Sectoral Indices | MF data",
                    "how_to_check": "Money into stock sector=Bullish",
                },
                "5.5 IPO Market": {
                    "sub_weight": 0.3, "default_score": 5,
                    "description": "Pipeline, listing performance",
                    "source": "chittorgarh.com/ipo/",
                    "how_to_check": "Healthy=Bullish, Excessive=Frothy",
                },
                "5.6 Global Contagion": {
                    "sub_weight": 0.7, "default_score": 5,
                    "description": "Correlation with global mkts",
                    "source": "S&P500, Hang Seng correlation",
                    "how_to_check": "Global stable=Bullish",
                },
                "5.7 Retail Participation": {
                    "sub_weight": 0.5, "default_score": 6,
                    "description": "Demat growth, retail AUM",
                    "source": "CDSL/NSDL data | SEBI",
                    "how_to_check": "Growing demats=Long-term Bullish",
                },
            },
        },
        "6. Structural (4%)": {
            "weight": 4.0,
            "factors": {
                "6.1 Demographics": {
                    "sub_weight": 0.7, "default_score": 7,
                    "description": "Working age, urbanization",
                    "source": "Census | UN | World Bank",
                    "how_to_check": "Young population=Structural Bullish",
                },
                "6.2 Digital Penetration": {
                    "sub_weight": 0.5, "default_score": 8,
                    "description": "UPI, internet users",
                    "source": "NPCI | TRAI | Statista",
                    "how_to_check": "Rising adoption=Bullish for tech",
                },
                "6.3 Monsoon & Agri": {
                    "sub_weight": 1.0, "default_score": 5,
                    "description": "IMD forecast, El Nino",
                    "source": "mausam.imd.gov.in",
                    "how_to_check": "Normal=Bullish rural, Deficient=Bearish",
                },
                "6.4 Real Estate Cycle": {
                    "sub_weight": 0.5, "default_score": 6,
                    "description": "Sales, inventory",
                    "source": "Knight Frank | JLL India",
                    "how_to_check": "Rising sales=Bullish RE/cement",
                },
                "6.5 Employment": {
                    "sub_weight": 0.5, "default_score": 5,
                    "description": "CMIE, EPFO data",
                    "source": "CMIE | EPFO | PLFS",
                    "how_to_check": "Falling unemployment=Bullish",
                },
                "6.6 Consumer Confidence": {
                    "sub_weight": 0.3, "default_score": 5,
                    "description": "RBI survey",
                    "source": "RBI Consumer Confidence Survey",
                    "how_to_check": "Improving=Bullish consumption",
                },
                "6.7 PMI": {
                    "sub_weight": 0.5, "default_score": 6,
                    "description": "Manufacturing & Services",
                    "source": "S&P Global India PMI",
                    "how_to_check": ">52=Bullish, 50-52=Neutral, <50=Bearish",
                },
            },
        },
    }

# ─────────────────────────────────────────────
# MICRO ANALYZER
# ─────────────────────────────────────────────

class MicroAnalyzer:
    def __init__(self, fetcher, sector):
        self.f = fetcher
        self.info = fetcher.info
        self.sector = sector

    def sg(self, key, default=None):
        return self.info.get(key, default)

    def analyze_financials(self):
        results = {}
        rg = self.sg("revenueGrowth")
        rs = 5
        if rg is not None:
            if rg > 0.20: rs = 9
            elif rg > 0.10: rs = 7
            elif rg > 0.05: rs = 5
            elif rg > 0: rs = 4
            else: rs = 2
        results["7.1 Revenue Growth"] = {
            "sub_weight": 2.5,
            "value": "{:.1f}%".format(rg * 100) if rg else "N/A",
            "score": rs,
            "source": "Yahoo Finance revenueGrowth",
            "how_to_check": "Screener.in > P&L > Revenue CAGR",
        }
    
        em = self.sg("ebitdaMargins")
        pm = self.sg("profitMargins")
        ps = 5
        if em is not None:
            if em > 0.25: ps = 9
            elif em > 0.15: ps = 7
            elif em > 0.10: ps = 5
            else: ps = 3
        val = "N/A"
        if em is not None and pm is not None:
            val = "EBITDA:{:.1f}% PAT:{:.1f}%".format(em*100, pm*100)
        results["7.2 Profitability"] = {
            "sub_weight": 2.5, "value": val, "score": ps,
            "source": "Yahoo Finance ebitdaMargins, profitMargins",
            "how_to_check": "Screener.in > Ratios",
        }
    
        eg = self.sg("earningsGrowth")
        es = 5
        if eg is not None:
            if eg > 0.25: es = 9
            elif eg > 0.15: es = 7
            elif eg > 0.05: es = 5
            elif eg > 0: es = 4
            else: es = 2
        results["7.3 EPS Growth"] = {
            "sub_weight": 1.5,
            "value": "{:.1f}%".format(eg*100) if eg else "N/A",
            "score": es,
            "source": "Yahoo Finance earningsGrowth",
            "how_to_check": "Screener.in > EPS trend",
        }
    
        roe = self.sg("returnOnEquity")
        roa = self.sg("returnOnAssets")
        rts = 5
        if roe is not None:
            if roe > 0.20: rts = 9
            elif roe > 0.15: rts = 7
            elif roe > 0.10: rts = 5
            else: rts = 3
        val2 = "N/A"
        if roe is not None and roa is not None:
            val2 = "ROE:{:.1f}% ROA:{:.1f}%".format(roe*100, roa*100)
        results["7.4 Return Ratios"] = {
            "sub_weight": 2.0, "value": val2, "score": rts,
            "source": "Yahoo Finance returnOnEquity, returnOnAssets",
            "how_to_check": "Screener.in > Ratios; ROCE from Tijori",
        }
    
        ocf = self.sg("operatingCashflow")
        fcf = self.sg("freeCashflow")
        cfs = 5
        if fcf is not None and ocf is not None:
            if fcf > 0 and ocf > 0:
                cfs = 7
                mc = self.sg("marketCap", 1)
                if mc and mc > 0:
                    fy = fcf / mc
                    if fy > 0.05: cfs = 9
                    elif fy > 0.03: cfs = 7
            elif ocf and ocf > 0:
                cfs = 5
            else:
                cfs = 2
        val3 = "N/A"
        if ocf is not None and fcf is not None:
            val3 = "OCF:Rs{:.0f}Cr FCF:Rs{:.0f}Cr".format(ocf/1e7, fcf/1e7)
        results["7.5 Cash Flow"] = {
            "sub_weight": 2.0, "value": val3, "score": cfs,
            "source": "Yahoo Finance operatingCashflow, freeCashflow",
            "how_to_check": "Screener.in > Cash Flow",
        }
    
        results["7.6 Working Capital"] = {
            "sub_weight": 1.0, "value": "Manual check", "score": 5,
            "source": "Manual",
            "how_to_check": "Screener.in > Debtor/Inventory Days",
        }
    
        de = self.sg("debtToEquity")
        cr = self.sg("currentRatio")
        bs = 5
        if de is not None:
            if de < 0.3: bs = 9
            elif de < 0.7: bs = 7
            elif de < 1.5: bs = 5
            else: bs = 3
        val4 = "N/A"
        if de is not None and cr is not None:
            val4 = "D/E:{:.2f} CR:{:.2f}".format(de, cr)
        results["7.7 Balance Sheet"] = {
            "sub_weight": 2.0, "value": val4, "score": bs,
            "source": "Yahoo Finance debtToEquity, currentRatio",
            "how_to_check": "Screener.in > Balance Sheet",
        }
    
        dy = self.sg("dividendYield")
        ds = 5
        if dy is not None:
            if dy > 0.03: ds = 8
            elif dy > 0.01: ds = 6
            elif dy > 0: ds = 4
            else: ds = 3
        results["7.8 Dividends"] = {
            "sub_weight": 0.5,
            "value": "{:.2f}%".format(dy*100) if dy else "N/A",
            "score": ds,
            "source": "Yahoo Finance dividendYield",
            "how_to_check": "Screener.in > Dividends; Tickertape",
        }
    
        results["7.9 Earnings Quality"] = {
            "sub_weight": 1.0, "value": "Check Annual Report", "score": 5,
            "source": "Annual Report - Auditor Report",
            "how_to_check": "Audit qualifications; Tofler.in",
        }
        return results

    def analyze_valuation(self):
        results = {}
        pet = self.sg("trailingPE")
        pef = self.sg("forwardPE")
        peg = self.sg("pegRatio")
        ps = 5
        if pef is not None and pet is not None:
            if pef < 15: ps = 8
            elif pef < 25: ps = 6
            elif pef < 40: ps = 4
            else: ps = 2
            if pef < pet:
                ps = min(10, ps + 1)
        val = "N/A"
        if pet is not None:
            val = "TTM:{:.1f}".format(pet)
            if pef is not None:
                val += " Fwd:{:.1f}".format(pef)
            if peg is not None:
                val += " PEG:{:.2f}".format(peg)
        results["8.1 PE Ratio"] = {
            "sub_weight": 1.5, "value": val, "score": ps,
            "source": "Yahoo Finance trailingPE, forwardPE, pegRatio",
            "how_to_check": "Screener.in PE chart; Tickertape Valuation",
        }
    
        eve = self.sg("enterpriseToEbitda")
        evs = 5
        if eve is not None:
            if eve < 10: evs = 8
            elif eve < 18: evs = 6
            elif eve < 30: evs = 4
            else: evs = 2
        results["8.2 EV/EBITDA"] = {
            "sub_weight": 1.5,
            "value": "{:.1f}".format(eve) if eve else "N/A",
            "score": evs,
            "source": "Yahoo Finance enterpriseToEbitda",
            "how_to_check": "Screener.in; Tijori Finance",
        }
    
        pb = self.sg("priceToBook")
        pbs = 5
        if pb is not None:
            if self.sector in ["banking", "realestate"]:
                if pb < 1.5: pbs = 8
                elif pb < 3.0: pbs = 6
                else: pbs = 3
            else:
                if pb < 3: pbs = 7
                elif pb < 6: pbs = 5
                else: pbs = 3
        results["8.3 Price/Book"] = {
            "sub_weight": 1.0,
            "value": "{:.2f}".format(pb) if pb else "N/A",
            "score": pbs,
            "source": "Yahoo Finance priceToBook",
            "how_to_check": "Screener.in > Ratios",
        }
    
        psr = self.sg("priceToSalesTrailing12Months")
        pss = 5
        if psr is not None:
            if psr < 2: pss = 8
            elif psr < 5: pss = 6
            elif psr < 10: pss = 4
            else: pss = 2
        results["8.4 Price/Sales"] = {
            "sub_weight": 0.5,
            "value": "{:.2f}".format(psr) if psr else "N/A",
            "score": pss,
            "source": "Yahoo Finance priceToSalesTrailing12Months",
            "how_to_check": "Tickertape Valuation",
        }
    
        results["8.5 DCF Value"] = {
            "sub_weight": 2.0, "value": "Manual DCF needed", "score": 5,
            "source": "Tickertape Fair Value; Trendlyne Forecaster",
            "how_to_check": "Build DCF or use Tickertape/Simply Wall St",
        }
    
        tp = self.sg("targetMeanPrice")
        cp = self.sg("currentPrice")
        ms = 5
        mv = "N/A"
        if tp and cp and cp > 0:
            up = (tp - cp) / cp
            mv = "Target:Rs{:.0f} CMP:Rs{:.0f} Up:{:.1f}%".format(tp, cp, up*100)
            if up > 0.30: ms = 9
            elif up > 0.15: ms = 7
            elif up > 0: ms = 5
            else: ms = 3
        results["8.6 Margin of Safety"] = {
            "sub_weight": 1.0, "value": mv, "score": ms,
            "source": "Yahoo Finance targetMeanPrice",
            "how_to_check": "Trendlyne Forecaster; MoneyControl Analysts",
        }
    
        results["8.7 Relative Valuation"] = {
            "sub_weight": 0.5, "value": "Compare with peers", "score": 5,
            "source": "Screener.in Peers; Tijori Finance",
            "how_to_check": "Screener.in > Peer Comparison",
        }
        return results

    def analyze_promoter(self):
        results = {}
        hip = self.sg("heldPercentInsiders")
        ps = 5
        if hip is not None:
            if hip > 0.60: ps = 8
            elif hip > 0.45: ps = 7
            elif hip > 0.30: ps = 5
            else: ps = 3
        results["9.1 Promoter Holding"] = {
            "sub_weight": 1.5,
            "value": "{:.1f}%".format(hip*100) if hip else "N/A",
            "score": ps,
            "source": "Yahoo Finance heldPercentInsiders",
            "how_to_check": "BSE Shareholding Pattern; Trendlyne",
        }
        quals = [
            ("9.2 Pledge %", 1.0, "BSE Shareholding Pattern"),
            ("9.3 Promoter Background", 1.5, "Annual Report Directors Profile"),
            ("9.4 Succession", 0.5, "Annual Report; News"),
            ("9.5 Compensation", 0.5, "Annual Report Governance"),
            ("9.6 Capital Allocation", 1.5, "Annual Report MD&A"),
            ("9.7 Governance Score", 1.5, "IiAS; Annual Report"),
            ("9.8 Insider Trading", 0.5, "BSE SAST Disclosures"),
            ("9.9 Entity Structure", 1.0, "Zauba Corp; Tofler.in"),
            ("9.10 Guidance Accuracy", 0.5, "Earnings call transcripts"),
        ]
        for name, w, src in quals:
            results[name] = {
                "sub_weight": w, "value": "Qualitative-check source",
                "score": 5, "source": src,
                "how_to_check": "See source reference",
            }
        return results
    
    def analyze_legal(self):
        results = {}
        items = [
            ("10.1 Civil Litigation", 0.8),
            ("10.2 Criminal Litigation", 0.8),
            ("10.3 Tax Disputes", 0.7),
            ("10.4 Regulatory Penalties", 0.5),
            ("10.5 Environmental Cases", 0.5),
            ("10.6 IP/Patent Disputes", 0.4),
            ("10.7 Labour Disputes", 0.3),
            ("10.8 Closed Litigation", 0.5),
            ("10.9 Fraud Allegations", 0.5),
        ]
        for name, w in items:
            results[name] = {
                "sub_weight": w,
                "value": "Qualitative-check Annual Report",
                "score": 5,
                "source": "Annual Report Contingent Liabilities; BSE",
                "how_to_check": "Annual Report Notes; Google News",
            }
        return results
    
    def analyze_competitive(self):
        results = {}
        ind = self.sg("industry", "N/A")
        items = [
            ("11.1 Market Share", 1.5, "Industry: " + str(ind)),
            ("11.2 Moat / Barriers", 1.5, "Qualitative"),
            ("11.3 Competition", 1.0, "Check peers"),
            ("11.4 Differentiation", 0.5, "Check products"),
            ("11.5 Customer Concentration", 0.5, "Annual Report"),
            ("11.6 Supplier Concentration", 0.5, "Annual Report"),
            ("11.7 Disruption Risk", 1.0, "Industry analysis"),
            ("11.8 Geographic Diversification", 0.5, "Segments"),
        ]
        for name, w, val in items:
            results[name] = {
                "sub_weight": w, "value": val, "score": 5,
                "source": "Investor presentation; Industry reports",
                "how_to_check": "Screener.in Peers; Company website",
            }
        return results
    
    def analyze_growth(self):
        results = {}
        rg = self.sg("revenueGrowth")
        eg = self.sg("earningsGrowth")
        items = [
            ("12.1 Order Book", 1.5, "Check investor pres"),
            ("12.2 Capacity Expansion", 1.0, "Check capex plans"),
            ("12.3 R&D Spend", 0.5, "Annual Report"),
            ("12.4 M&A Strategy", 0.5, "News; BSE announcements"),
            ("12.5 New Markets", 0.5, "Investor presentation"),
            ("12.6 Digital Transformation", 0.5, "Annual Report"),
        ]
        for name, w, val in items:
            results[name] = {
                "sub_weight": w, "value": val, "score": 5,
                "source": "Investor presentation; Annual Report",
                "how_to_check": "Trendlyne Earnings Calls",
            }
        gv = "Check earnings call"
        gs = 5
        if rg is not None and eg is not None:
            gv = "Rev:{:.1f}% Earn:{:.1f}%".format(rg*100, eg*100)
            if rg > 0.1:
                gs = 6
        results["12.7 Guidance"] = {
            "sub_weight": 1.0, "value": gv, "score": gs,
            "source": "Earnings calls; Analyst reports",
            "how_to_check": "Trendlyne; Screener.in Con Calls",
        }
        results["12.8 Industry Tailwinds"] = {
            "sub_weight": 1.0, "value": "Sector assessment", "score": 5,
            "source": "IBEF; CRISIL outlook",
            "how_to_check": "Google India [sector] outlook",
        }
        results["12.9 ESG"] = {
            "sub_weight": 0.5, "value": "Check BRSR", "score": 5,
            "source": "BRSR Report; Sustainalytics",
            "how_to_check": "Annual Report BRSR section",
        }
        return results
    
    def analyze_shareholding(self):
        results = {}
        fp = self.sg("heldPercentInstitutions")
        fs = 5
        if fp is not None:
            if fp > 0.40: fs = 8
            elif fp > 0.25: fs = 7
            elif fp > 0.10: fs = 5
            else: fs = 3
        results["13.1 FII Holding"] = {
            "sub_weight": 0.8,
            "value": "Inst:{:.1f}%".format(fp*100) if fp else "N/A",
            "score": fs,
            "source": "Yahoo Finance heldPercentInstitutions",
            "how_to_check": "BSE Shareholding; Trendlyne",
        }
        results["13.2 DII/MF Holding"] = {
            "sub_weight": 0.8, "value": "Check BSE", "score": 5,
            "source": "BSE Shareholding; AMFI",
            "how_to_check": "Trendlyne MF Holding",
        }
        results["13.3 Retail Mix"] = {
            "sub_weight": 0.3, "value": "Check pattern", "score": 5,
            "source": "BSE Shareholding",
            "how_to_check": "High retail>30%=volatile",
        }
        results["13.4 Bulk Deals"] = {
            "sub_weight": 0.3, "value": "Check NSE/BSE", "score": 5,
            "source": "NSE/BSE Bulk Deals",
            "how_to_check": "Trendlyne Bulk Deals",
        }
        na = self.sg("numberOfAnalystOpinions")
        tp = self.sg("targetMeanPrice")
        rk = self.sg("recommendationKey", "N/A")
        ans = 5
        if na and na > 15: ans = 7
        elif na and na > 5: ans = 6
        av = "N/A"
        if na and tp:
            av = "Analysts:{} Con:{} Tgt:Rs{:.0f}".format(na, rk, tp)
        results["13.5 Analyst Coverage"] = {
            "sub_weight": 0.4, "value": av, "score": ans,
            "source": "Yahoo Finance analyst data",
            "how_to_check": "Trendlyne Forecaster; MoneyControl",
        }
        results["13.6 Index Inclusion"] = {
            "sub_weight": 0.4, "value": "Check niftyindices.com", "score": 5,
            "source": "niftyindices.com",
            "how_to_check": "Nifty50/Next50/MSCI India",
        }
        results["13.7 Short Interest"] = {
            "sub_weight": 0.5, "value": "Check NSE F&O", "score": 5,
            "source": "NSE Derivatives OI",
            "how_to_check": "NSE > Derivatives > OI data",
        }
        fls = self.sg("floatShares")
        results["13.8 Free Float"] = {
            "sub_weight": 0.5,
            "value": "Float:{:,.0f}".format(fls) if fls else "N/A",
            "score": 5,
            "source": "Yahoo Finance floatShares",
            "how_to_check": "BSE Public holding %",
        }
        return results
    
    def analyze_technicals(self):
        results = {}
        hist = self.f.history
        if hist.empty or len(hist) < 50:
            for n in ["14.1 Trend", "14.2 Volume", "14.3 RSI/MACD",
                       "14.4 Support/Resistance", "14.5 52W Proximity",
                       "14.6 Rel Strength", "14.7 Patterns",
                       "14.8 Volatility", "14.9 Delivery"]:
                results[n] = {
                    "sub_weight": 0.5, "value": "Insufficient data",
                    "score": 5, "source": "N/A",
                    "how_to_check": "Need 50+ days data",
                }
            return results
    
        close = hist["Close"]
        volume = hist["Volume"]
        cp = close.iloc[-1]
    
        s50 = close.rolling(50).mean().iloc[-1]
        s200 = None
        if len(close) >= 200:
            s200 = close.rolling(200).mean().iloc[-1]
        ts = 5
        tt = "CMP:Rs{:.1f} 50D:Rs{:.1f}".format(cp, s50)
        if s200 is not None:
            tt += " 200D:Rs{:.1f}".format(s200)
            if cp > s50 > s200:
                ts = 8
                tt += " BULLISH"
            elif cp > s200:
                ts = 6
                tt += " MOD BULLISH"
            elif cp < s50 < s200:
                ts = 2
                tt += " BEARISH"
            else:
                ts = 4
                tt += " MIXED"
        results["14.1 Trend"] = {
            "sub_weight": 0.7, "value": tt, "score": ts,
            "source": "Calculated 50/200 DMA",
            "how_to_check": "TradingView.com",
        }
    
        av20 = volume.tail(20).mean()
        av50 = volume.tail(50).mean()
        vs = 5
        vt = "20D:{:,.0f} 50D:{:,.0f}".format(av20, av50)
        if av20 > av50 * 1.3:
            vs = 7
            vt += " EXPANDING"
        elif av20 < av50 * 0.7:
            vs = 4
            vt += " CONTRACTING"
        results["14.2 Volume"] = {
            "sub_weight": 0.5, "value": vt, "score": vs,
            "source": "Yahoo Finance volume",
            "how_to_check": "TradingView; ChartInk",
        }
    
        rsi_s = 5
        rsi_t = "N/A"
        if TA_AVAILABLE:
            try:
                ri = ta_lib.momentum.RSIIndicator(close, window=14)
                rsi_val = ri.rsi().iloc[-1]
                mi = ta_lib.trend.MACD(close)
                ml = mi.macd().iloc[-1]
                ms_val = mi.macd_signal().iloc[-1]
                rsi_t = "RSI:{:.1f} MACD:{:.2f} Sig:{:.2f}".format(
                    rsi_val, ml, ms_val
                )
                if rsi_val < 30:
                    rsi_s = 8
                    rsi_t += " OVERSOLD"
                elif rsi_val < 40:
                    rsi_s = 6
                elif rsi_val > 70:
                    rsi_s = 3
                    rsi_t += " OVERBOUGHT"
                elif rsi_val > 60:
                    rsi_s = 5
                else:
                    rsi_s = 5
                if ml > ms_val:
                    rsi_s = min(10, rsi_s + 1)
                    rsi_t += " MACD+"
            except Exception:
                rsi_t = "Calculation error"
        results["14.3 RSI/MACD"] = {
            "sub_weight": 0.4, "value": rsi_t, "score": rsi_s,
            "source": "ta library RSI-14, MACD 12-26-9",
            "how_to_check": "TradingView Indicators",
        }
    
        h52 = close.tail(252).max() if len(close) >= 252 else close.max()
        l52 = close.tail(252).min() if len(close) >= 252 else close.min()
        results["14.4 Support/Resistance"] = {
            "sub_weight": 0.4,
            "value": "52H:Rs{:.1f} 52L:Rs{:.1f} CMP:Rs{:.1f}".format(
                h52, l52, cp
            ),
            "score": 5,
            "source": "Yahoo Finance 52W data",
            "how_to_check": "TradingView; NSE",
        }
    
        rng = h52 - l52
        pxs = 5
        pxt = "N/A"
        if rng > 0:
            pos = (cp - l52) / rng
            if pos > 0.9:
                pxs = 4
                pxt = "Near 52W High ({:.0f}%)".format(pos*100)
            elif pos > 0.6:
                pxs = 6
                pxt = "Upper half ({:.0f}%)".format(pos*100)
            elif pos > 0.3:
                pxs = 7
                pxt = "Mid-range ({:.0f}%)".format(pos*100)
            else:
                pxs = 5
                pxt = "Near 52W Low ({:.0f}%)".format(pos*100)
        results["14.5 52W Proximity"] = {
            "sub_weight": 0.3, "value": pxt, "score": pxs,
            "source": "Calculated from 52W range",
            "how_to_check": "NSE/BSE 52W H/L",
        }
    
        beta = self.sg("beta")
        brs = 5
        results["14.6 Rel Strength"] = {
            "sub_weight": 0.4,
            "value": "Beta:{:.2f}".format(beta) if beta else "N/A",
            "score": brs,
            "source": "Yahoo Finance beta",
            "how_to_check": "Tickertape; StockEdge",
        }
    
        results["14.7 Patterns"] = {
            "sub_weight": 0.3, "value": "Visual analysis needed",
            "score": 5, "source": "TradingView",
            "how_to_check": "TradingView chart tools",
        }
    
        rets = close.pct_change().dropna()
        vol_ann = rets.std() * np.sqrt(252) * 100
        vls = 5
        if vol_ann < 25: vls = 7
        elif vol_ann < 40: vls = 5
        else: vls = 3
        vv = "AnnVol:{:.1f}%".format(vol_ann)
        if beta:
            vv += " Beta:{:.2f}".format(beta)
        results["14.8 Volatility"] = {
            "sub_weight": 0.5, "value": vv, "score": vls,
            "source": "Calculated annualized std dev",
            "how_to_check": "Tickertape Risk metrics",
        }
    
        results["14.9 Delivery"] = {
            "sub_weight": 0.5, "value": "Check NSE", "score": 5,
            "source": "NSE Delivery Position data",
            "how_to_check": "NSE > Market Data > Delivery; Chartink",
        }
        return results
    
    # ─────────────────────────────────────────────
    # SECTOR ADJUSTMENTS
    # ─────────────────────────────────────────────

    def get_sector_adjustments(sector):
        adj = {
            "banking": {
                "NPA/Asset Quality": {"weight": 3.0, "score": 5, "source": "Quarterly results; RBI DBIE", "how_to_check": "GNPA<3%=Good"},
                "NIM/Spread": {"weight": 2.0, "score": 5, "source": "Investor presentation", "how_to_check": "NIM>3%=Good"},
                "CASA Ratio": {"weight": 1.0, "score": 5, "source": "Quarterly results", "how_to_check": "CASA>40%=Good"},
                "Capital Adequacy": {"weight": 1.5, "score": 5, "source": "Basel III disclosure", "how_to_check": "CET1>10%=Comfortable"},
            },
            "it": {
                "Attrition Rate": {"weight": 1.5, "score": 5, "source": "Quarterly results HR section", "how_to_check": "LTM<15%=Good"},
                "Deal Wins/TCV": {"weight": 2.0, "score": 5, "source": "Earnings release", "how_to_check": "TCV growing QoQ=healthy"},
                "Utilization Rate": {"weight": 1.0, "score": 5, "source": "Investor presentation", "how_to_check": ">82%=Good"},
            },
            "pharma": {
                "USFDA Actions": {"weight": 3.0, "score": 5, "source": "fda.gov Inspections", "how_to_check": "Warning Letters, 483s"},
                "ANDA Pipeline": {"weight": 2.0, "score": 5, "source": "FDA Orange Book; Inv Pres", "how_to_check": "ANDAs filed/approved"},
                "Price Control": {"weight": 1.5, "score": 5, "source": "NPPA; NLEM list", "how_to_check": "% revenue under DPCO"},
            },
            "energy": {
                "Gas Prices": {"weight": 2.0, "score": 5, "source": "PPAC India", "how_to_check": "APM pricing"},
                "GRM": {"weight": 2.0, "score": 5, "source": "Singapore GRM benchmark", "how_to_check": "Quarterly GRM trend"},
                "Subsidy": {"weight": 1.5, "score": 5, "source": "PPAC; Budget", "how_to_check": "Fuel pricing freedom"},
            },
            "fmcg": {
                "RM Costs": {"weight": 1.5, "score": 5, "source": "MCX; quarterly results", "how_to_check": "Palm oil, milk trends"},
                "Distribution": {"weight": 1.5, "score": 5, "source": "Investor presentation", "how_to_check": ">1M outlets=Strong"},
                "Volume vs Price": {"weight": 1.0, "score": 5, "source": "Earnings commentary", "how_to_check": "Vol>5%=healthy"},
            },
            "auto": {
                "RM Prices": {"weight": 1.5, "score": 5, "source": "LME; MCX", "how_to_check": "Steel/aluminum trend"},
                "EV Transition": {"weight": 2.0, "score": 5, "source": "SIAM/FADA; Company EV strategy", "how_to_check": "EV portfolio %"},
                "Monthly Sales": {"weight": 1.0, "score": 5, "source": "siam.in; fada.in", "how_to_check": "Dispatch & retail YoY"},
            },
            "metals": {
                "China Demand": {"weight": 2.0, "score": 5, "source": "China PMI; property data", "how_to_check": "China steel production"},
                "Mining Leases": {"weight": 1.0, "score": 5, "source": "MMDR Act; Annual Report", "how_to_check": "Lease expiry dates"},
            },
            "realestate": {
                "RERA Compliance": {"weight": 1.5, "score": 5, "source": "State RERA websites", "how_to_check": "Delivery track record"},
                "Land Bank": {"weight": 1.5, "score": 5, "source": "Investor presentation", "how_to_check": "Sq ft; location quality"},
                "Pre-Sales": {"weight": 1.5, "score": 5, "source": "Quarterly pre-sales data", "how_to_check": "Growing QoQ=strong"},
            },
            "telecom": {
                "ARPU Trend": {"weight": 2.0, "score": 5, "source": "Quarterly results; TRAI", "how_to_check": ">Rs200=Good"},
                "Subscribers": {"weight": 1.0, "score": 5, "source": "TRAI data", "how_to_check": "Net adds trend"},
                "Spectrum Costs": {"weight": 2.0, "score": 5, "source": "DoT auction results", "how_to_check": "Payment obligations"},
            },
            "power": {
                "Tariff Orders": {"weight": 2.5, "score": 5, "source": "CERC/SERC orders", "how_to_check": "Allowed ROE"},
                "PLF/CUF": {"weight": 1.0, "score": 5, "source": "CEA data", "how_to_check": "Thermal PLF>70%=Good"},
                "Coal Supply": {"weight": 1.5, "score": 5, "source": "CIL data", "how_to_check": "Coal stock days"},
            },
            "chemicals": {
                "China+1": {"weight": 2.0, "score": 5, "source": "Industry reports", "how_to_check": "Supply shift wins"},
                "Registrations": {"weight": 1.0, "score": 5, "source": "REACH/EPA status", "how_to_check": "Products registered"},
            },
            "cement": {
                "Limestone Reserves": {"weight": 1.0, "score": 5, "source": "Annual Report Mining", "how_to_check": ">30yr=comfortable"},
                "Power/Fuel Cost": {"weight": 1.5, "score": 5, "source": "Cost breakup", "how_to_check": "Fuel cost/ton trend"},
                "EBITDA/Ton": {"weight": 1.5, "score": 5, "source": "Investor presentation", "how_to_check": ">Rs1000=Good"},
            },
            "defence": {
                "Defence Budget": {"weight": 3.0, "score": 6, "source": "Union Budget", "how_to_check": "Capital outlay growth"},
                "Make in India": {"weight": 2.0, "score": 6, "source": "MoD Positive Lists", "how_to_check": "Product on positive list?"},
            },
            "textiles": {
                "Cotton Prices": {"weight": 2.0, "score": 5, "source": "MCX Cotton; CAI", "how_to_check": "Stable=margin benefit"},
                "FTAs": {"weight": 1.5, "score": 5, "source": "Commerce Ministry", "how_to_check": "India-EU/UK FTA progress"},
            },
        }
        return adj.get(sector, {})
    
    # ─────────────────────────────────────────────
    # SCORE CALCULATOR
    # ─────────────────────────────────────────────
    
    def calculate_composite(macro_scores, micro_sections, sector_adj):
        macro_tw = 0
        macro_ws = 0
        macro_bk = {}
        for cat, data in macro_scores.items():
            cw = data["weight"]
            css = 0
            csw = 0
            for f, fd in data["factors"].items():
                sw = fd["sub_weight"]
                sc = fd.get("score", fd["default_score"])
                css += sw * sc
                csw += sw
            ca = css / csw if csw > 0 else 5
            macro_bk[cat] = {"weight": cw, "avg_score": ca}
            macro_ws += cw * ca
            macro_tw += cw
        m10 = macro_ws / macro_tw if macro_tw > 0 else 5
        
        micro_tw = 0
        micro_ws = 0
        mw = {
            "7_fin": 15, "8_val": 8, "9_prom": 10, "10_leg": 5,
            "11_comp": 7, "12_grow": 7, "13_share": 4, "14_tech": 4,
        }
        micro_bk = {}
        for sk, sd in micro_sections.items():
            sw_total = mw.get(sk, 5)
            sss = 0
            ssw = 0
            for f, fd in sd.items():
                sss += fd["sub_weight"] * fd["score"]
                ssw += fd["sub_weight"]
            sa = sss / ssw if ssw > 0 else 5
            micro_bk[sk] = {"weight": sw_total, "avg_score": sa}
            micro_ws += sw_total * sa
            micro_tw += sw_total
        mi10 = micro_ws / micro_tw if micro_tw > 0 else 5
        
        sb = 0
        for f, fd in sector_adj.items():
            sb += fd["weight"] * (fd["score"] - 5) / 10
        
        base = (0.4 * m10 + 0.6 * mi10) * 10
        final = max(0, min(100, base + sb))
        return {
            "macro_score_10": m10, "micro_score_10": mi10,
            "macro_breakdown": macro_bk, "micro_breakdown": micro_bk,
            "sector_bonus": sb, "base_score": base, "final_score": final,
        }
        
        def get_recommendation(score):
            for (lo, hi), (rec, em) in SCORE_INTERPRETATION.items():
                if lo <= score < hi:
                    return rec, em
            return "N/A", "?"
        
    # ─────────────────────────────────────────────
    # EXCEL EXPORT
    # ─────────────────────────────────────────────
    
    def generate_excel(stock_results):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            wb = writer.book
            hf = wb.add_format({
                "bold": True, "bg_color": "#1F4E79",
                "font_color": "white", "border": 1,
                "text_wrap": True, "valign": "vcenter",
            })
            sg = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
            sy = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
            sr = wb.add_format({"bg_color": "#FFC7CE", "border": 1})
        
            for sn, res in stock_results.items():
                safe = sn.replace("/", "_")[:25]
        
                # Summary
                sdata = {
                    "Metric": [
                        "Stock", "Sector", "Final Score", "Recommendation",
                        "Macro (of 10)", "Micro (of 10)", "Sector Adj",
                        "CMP", "Market Cap", "PE TTM", "PE Fwd",
                        "ROE", "D/E", "52W High", "52W Low", "Beta",
                    ],
                    "Value": [
                        sn, res.get("sector", ""),
                        "{:.1f}/100".format(res["composite"]["final_score"]),
                        res["recommendation"][0],
                        "{:.2f}".format(res["composite"]["macro_score_10"]),
                        "{:.2f}".format(res["composite"]["micro_score_10"]),
                        "{:.2f}".format(res["composite"]["sector_bonus"]),
                        str(res["info"].get("currentPrice", "N/A")),
                        str(res["info"].get("marketCap", "N/A")),
                        str(res["info"].get("trailingPE", "N/A")),
                        str(res["info"].get("forwardPE", "N/A")),
                        str(res["info"].get("returnOnEquity", "N/A")),
                        str(res["info"].get("debtToEquity", "N/A")),
                        str(res["info"].get("fiftyTwoWeekHigh", "N/A")),
                        str(res["info"].get("fiftyTwoWeekLow", "N/A")),
                        str(res["info"].get("beta", "N/A")),
                    ],
                }
                pd.DataFrame(sdata).to_excel(
                    writer, sheet_name=safe + "_Sum", index=False
                )
                ws = writer.sheets[safe + "_Sum"]
                ws.set_column("A:A", 22)
                ws.set_column("B:B", 30)
        
                # Detail
                af = []
                for cat, cd in res["macro_data"].items():
                    for fac, fd in cd["factors"].items():
                        af.append({
                            "Section": "MACRO", "Category": cat,
                            "Factor": fac,
                            "Weight%": fd["sub_weight"],
                            "Score": fd.get("score", fd["default_score"]),
                            "Value": fd.get("description", ""),
                            "Source": fd.get("source", ""),
                            "How to Check": fd.get("how_to_check", ""),
                        })
                for sk, sd in res["micro_sections"].items():
                    for fac, fd in sd.items():
                        af.append({
                            "Section": "MICRO", "Category": sk,
                            "Factor": fac,
                            "Weight%": fd["sub_weight"],
                            "Score": fd["score"],
                            "Value": fd.get("value", ""),
                            "Source": fd.get("source", ""),
                            "How to Check": fd.get("how_to_check", ""),
                        })
                for fac, fd in res.get("sector_adj", {}).items():
                    af.append({
                        "Section": "SECTOR", "Category": res.get("sector", ""),
                        "Factor": fac,
                        "Weight%": fd["weight"],
                        "Score": fd["score"],
                        "Value": "",
                        "Source": fd.get("source", ""),
                        "How to Check": fd.get("how_to_check", ""),
                    })
                df = pd.DataFrame(af)
                df.to_excel(writer, sheet_name=safe + "_Det", index=False)
                ws2 = writer.sheets[safe + "_Det"]
                ws2.set_column("A:A", 10)
                ws2.set_column("B:B", 30)
                ws2.set_column("C:C", 35)
                ws2.set_column("D:D", 8)
                ws2.set_column("E:E", 6)
                ws2.set_column("F:F", 40)
                ws2.set_column("G:G", 50)
                ws2.set_column("H:H", 50)
                for ci, cv in enumerate(df.columns):
                    ws2.write(0, ci, cv, hf)
                nr = len(df)
                ws2.conditional_format(1, 4, nr, 4, {
                    "type": "cell", "criteria": ">=", "value": 7, "format": sg,
                })
                ws2.conditional_format(1, 4, nr, 4, {
                    "type": "cell", "criteria": "between",
                    "minimum": 4, "maximum": 6, "format": sy,
                })
                ws2.conditional_format(1, 4, nr, 4, {
                    "type": "cell", "criteria": "<", "value": 4, "format": sr,
                })
        output.seek(0)
        return output
        
    # ─────────────────────────────────────────────
    # MAIN APP
    # ─────────────────────────────────────────────
    
    def main():
        st.title("📊 Indian Stock Analyzer")
        st.markdown("**150+ Factor Framework for Indian Equities**")
        st.markdown("---")
        
        with st.sidebar:
            st.header("Configuration")
            st.markdown("**Enter NSE Symbols** (e.g. RELIANCE, TCS)")
            stock_input = st.text_area(
                "Stocks (one per line or comma-separated)",
                value="RELIANCE\nTCS", height=100,
            )
            sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
            analyze = st.button(
                "🔍 ANALYZE", type="primary", use_container_width=True
            )
            st.markdown("---")
            st.markdown("""
        **Instructions:**
        1. Enter NSE symbols
        2. Pick sector
        3. Click ANALYZE
        4. Download Excel
        5. Update qualitative scores in Excel
        
        *Qualitative factors default to 5/10*
            """)
        
        stocks = []
        if stock_input:
            for s in stock_input.replace(",", "\n").split("\n"):
                s = s.strip().upper()
                if s:
                    stocks.append(s)
        
        if analyze and stocks:
            sk = SECTOR_MAP[sector]
            all_res = {}
            prog = st.progress(0)
            status = st.empty()
        
            for idx, sn in enumerate(stocks):
                status.markdown("Analyzing **{}** ({}/{})...".format(
                    sn, idx + 1, len(stocks)
                ))
                fetcher = StockDataFetcher(sn)
                ok = fetcher.fetch_all()
                if not ok:
                    st.error("Could not fetch " + sn)
                    continue
        
                macro = get_macro_defaults()
                mi = MicroAnalyzer(fetcher, sk)
                micro = {
                    "7_fin": mi.analyze_financials(),
                    "8_val": mi.analyze_valuation(),
                    "9_prom": mi.analyze_promoter(),
                    "10_leg": mi.analyze_legal(),
                    "11_comp": mi.analyze_competitive(),
                    "12_grow": mi.analyze_growth(),
                    "13_share": mi.analyze_shareholding(),
                    "14_tech": mi.analyze_technicals(),
                }
                sadj = get_sector_adjustments(sk)
                comp = calculate_composite(macro, micro, sadj)
                rec = get_recommendation(comp["final_score"])
        
                all_res[sn] = {
                    "sector": sector, "info": fetcher.info,
                    "macro_data": macro, "micro_sections": micro,
                    "sector_adj": sadj, "composite": comp,
                    "recommendation": rec,
                    "data_sources": fetcher.data_sources,
                }
                prog.progress((idx + 1) / len(stocks))
        
            status.markdown("### ✅ Analysis Complete!")
        
            if all_res:
                st.markdown("---")
                st.header("Summary")
                cols = st.columns(min(len(all_res), 4))
                for i, (stk, res) in enumerate(all_res.items()):
                    with cols[i % 4]:
                        sc = res["composite"]["final_score"]
                        rc, em = res["recommendation"]
                        if sc >= 70:
                            cl = "#28a745"
                        elif sc >= 55:
                            cl = "#ffc107"
                        else:
                            cl = "#dc3545"
                        st.markdown(
                            '<div style="border:2px solid {c};border-radius:10px;'
                            'padding:15px;text-align:center;margin:5px">'
                            '<h3>{s}</h3><h1 style="color:{c}">{sc:.1f}</h1>'
                            '<h4>{e} {r}</h4>'
                            '<p>Macro:{m:.1f} Micro:{mi:.1f}</p></div>'.format(
                                c=cl, s=stk, sc=sc, e=em, r=rc,
                                m=res["composite"]["macro_score_10"],
                                mi=res["composite"]["micro_score_10"],
                            ),
                            unsafe_allow_html=True,
                        )
        
                for sn, res in all_res.items():
                    st.markdown("---")
                    st.header("📈 " + sn)
                    t1, t2, t3, t4, t5 = st.tabs([
                        "Overview", "Macro", "Micro", "Sector", "Sources"
                    ])
        
                    with t1:
                        info = res["info"]
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("CMP", "Rs{}".format(info.get("currentPrice", "N/A")))
                            mc = info.get("marketCap", 0)
                            st.metric("Mkt Cap", "Rs{:,.0f}Cr".format(mc / 1e7) if mc else "N/A")
                        with c2:
                            st.metric("PE TTM", str(info.get("trailingPE", "N/A")))
                            st.metric("PE Fwd", str(info.get("forwardPE", "N/A")))
                        with c3:
                            rv = info.get("returnOnEquity")
                            st.metric("ROE", "{:.1f}%".format(rv * 100) if rv else "N/A")
                            dv = info.get("debtToEquity")
                            st.metric("D/E", "{:.2f}".format(dv) if dv else "N/A")
                        with c4:
                            st.metric("52W Hi", "Rs{}".format(info.get("fiftyTwoWeekHigh", "N/A")))
                            st.metric("52W Lo", "Rs{}".format(info.get("fiftyTwoWeekLow", "N/A")))
        
                        mbk = res["composite"]["micro_breakdown"]
                        labels = {
                            "7_fin": "Financials", "8_val": "Valuation",
                            "9_prom": "Promoter", "10_leg": "Legal",
                            "11_comp": "Competition", "12_grow": "Growth",
                            "13_share": "Shareholding", "14_tech": "Technicals",
                        }
                        cats = list(mbk.keys())
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[labels.get(c, c) for c in cats],
                            y=[mbk[c]["avg_score"] for c in cats],
                            marker_color=[
                                "#2E86AB" if mbk[c]["avg_score"] >= 6
                                else "#F6AE2D" if mbk[c]["avg_score"] >= 4
                                else "#E94F37" for c in cats
                            ],
                            text=["{:.1f}".format(mbk[c]["avg_score"]) for c in cats],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title="Micro Scores - " + sn,
                            yaxis_title="Score (of 10)",
                            yaxis_range=[0, 10], height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
                    with t2:
                        st.subheader("Macro Scores (defaults-update in Excel)")
                        for cat, cd in res["macro_data"].items():
                            with st.expander(cat, expanded=False):
                                rows = []
                                for fac, fd in cd["factors"].items():
                                    rows.append({
                                        "Factor": fac,
                                        "Wt%": fd["sub_weight"],
                                        "Score": fd.get("score", fd["default_score"]),
                                        "Desc": fd["description"],
                                        "Source": fd["source"],
                                        "Check": fd["how_to_check"],
                                    })
                                st.dataframe(
                                    pd.DataFrame(rows),
                                    use_container_width=True, hide_index=True,
                                )
        
                    with t3:
                        st.subheader("Micro Scores")
                        snames = {
                            "7_fin": "7. Financials (15%)",
                            "8_val": "8. Valuation (8%)",
                            "9_prom": "9. Promoter (10%)",
                            "10_leg": "10. Legal (5%)",
                            "11_comp": "11. Competition (7%)",
                            "12_grow": "12. Growth (7%)",
                            "13_share": "13. Shareholding (4%)",
                            "14_tech": "14. Technicals (4%)",
                        }
                        for sk2, sd in res["micro_sections"].items():
                            with st.expander(snames.get(sk2, sk2), expanded=False):
                                rows = []
                                for fac, fd in sd.items():
                                    rows.append({
                                        "Factor": fac,
                                        "Wt%": fd["sub_weight"],
                                        "Score": fd["score"],
                                        "Value": fd.get("value", ""),
                                        "Source": fd.get("source", ""),
                                        "Check": fd.get("how_to_check", ""),
                                    })
                                st.dataframe(
                                    pd.DataFrame(rows),
                                    use_container_width=True, hide_index=True,
                                )
        
                    with t4:
                        st.subheader("Sector: " + sector)
                        if res["sector_adj"]:
                            rows = []
                            for fac, fd in res["sector_adj"].items():
                                rows.append({
                                    "Factor": fac,
                                    "Wt%": fd["weight"],
                                    "Score": fd["score"],
                                    "Source": fd.get("source", ""),
                                    "Check": fd.get("how_to_check", ""),
                                })
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True, hide_index=True,
                            )
                            st.info("Defaults to 5. Update in Excel.")
                        else:
                            st.info("No sector adjustments for Other.")
        
                    with t5:
                        st.subheader("Data Sources")
                        for k, v in res["data_sources"].items():
                            st.markdown("- **{}**: {}".format(k, v))
                        st.markdown("""
        ### Key Indian Sources
        | Source | URL | Use |
        |--------|-----|-----|
        | Screener.in | screener.in | Financials, ratios, peers |
        | Trendlyne | trendlyne.com | Shareholding, forecaster |
        | Tickertape | tickertape.in | Valuation, risk |
        | Tijori | tijorifinance.com | Deep financials |
        | MoneyControl | moneycontrol.com | News, analysts |
        | BSE | bseindia.com | Shareholding, filings |
        | NSE | nseindia.com | F&O, delivery data |
        | TradingView | tradingview.com | Charts, technicals |
        | Chartink | chartink.com | Screeners |
        | IBEF | ibef.org | Industry reports |
                        """)
        
                st.markdown("---")
                st.header("📥 Download Excel")
                xl = generate_excel(all_res)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="📥 Download Report",
                    data=xl,
                    file_name="Analysis_{}_{}.xlsx".format("_".join(stocks), ts),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )
                st.info(
                    "Update qualitative scores in Excel for accuracy. "
                    "All sources and instructions included."
                )
        
        elif analyze and not stocks:
            st.warning("Enter at least one stock symbol.")
        
        st.markdown("---")
        st.caption(
            "Disclaimer: Educational/research only. "
            "Not financial advice. Consult SEBI-registered advisor."
        )

    main()
