"""
Comprehensive Indian Stock Analysis Tool
Based on the 150+ Factor Framework (India Focus)
"""

import streamlit as st
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
import ta as ta_lib

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

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
(85, 100): ("STRONG BUY", "🟢"),
(70, 84): ("BUY", "🟡"),
(55, 69): ("HOLD", "🟠"),
(40, 54): ("UNDERPERFORM", "🔴"),
(0, 39): ("SELL / AVOID", "⛔"),
}

def get_yahoo_ticker(stock_name):
"""Convert stock name to Yahoo Finance ticker format."""
stock_name = stock_name.strip().upper()
if not stock_name.endswith(".NS") and not stock_name.endswith(".BO"):
    return stock_name + ".NS"
return stock_name

# =============================================================================
# DATA FETCHING
# =============================================================================

class StockDataFetcher:
"""Fetches all available data for a given stock."""

def __init__(self, ticker_symbol):
    self.symbol = ticker_symbol
    self.yahoo_ticker = get_yahoo_ticker(ticker_symbol)
    self.stock = None
    self.info = {}
    self.history = pd.DataFrame()
    self.financials = pd.DataFrame()
    self.balance_sheet = pd.DataFrame()
    self.cashflow = pd.DataFrame()
    self.quarterly_financials = pd.DataFrame()
    self.holders = {}
    self.recommendations = pd.DataFrame()
    self.data_sources = {}

def fetch_all(self):
    """Fetch all available data."""
    try:
        self.stock = yf.Ticker(self.yahoo_ticker)
        self.info = self.stock.info or {}

        self.history = self.stock.history(period="2y")
        self.data_sources["Price History"] = (
            "Yahoo Finance (" + self.yahoo_ticker + ") - 2 year daily data"
        )

        try:
            self.financials = self.stock.financials
            self.data_sources["Income Statement"] = (
                "Yahoo Finance - Annual Financials"
            )
        except Exception:
            self.financials = pd.DataFrame()

        try:
            self.quarterly_financials = self.stock.quarterly_financials
            self.data_sources["Quarterly Financials"] = (
                "Yahoo Finance - Quarterly Results"
            )
        except Exception:
            self.quarterly_financials = pd.DataFrame()

        try:
            self.balance_sheet = self.stock.balance_sheet
            self.data_sources["Balance Sheet"] = (
                "Yahoo Finance - Annual Balance Sheet"
            )
        except Exception:
            self.balance_sheet = pd.DataFrame()

        try:
            self.cashflow = self.stock.cashflow
            self.data_sources["Cash Flow"] = (
                "Yahoo Finance - Annual Cash Flow"
            )
        except Exception:
            self.cashflow = pd.DataFrame()

        try:
            self.holders = {
                "major": self.stock.major_holders,
                "institutional": self.stock.institutional_holders,
                "mutual_fund": self.stock.mutualfund_holders,
            }
            self.data_sources["Shareholding"] = (
                "Yahoo Finance - Holders Data"
            )
        except Exception:
            self.holders = {}

        try:
            self.recommendations = self.stock.recommendations
            self.data_sources["Analyst Recommendations"] = (
                "Yahoo Finance - Analyst Consensus"
            )
        except Exception:
            self.recommendations = pd.DataFrame()

        self.data_sources["Company Info"] = (
            "Yahoo Finance - " + self.yahoo_ticker + " info endpoint"
        )
        self.data_sources["Technical Indicators"] = (
            "Calculated from Yahoo Finance price history using ta library"
        )

        return True
    except Exception as e:
        st.error("Error fetching data for " + self.symbol + ": " + str(e))
        return False

# =============================================================================
# MACRO ANALYZER
# =============================================================================

class MacroAnalyzer:
"""Macro factors are market-wide and do not change per stock."""

@staticmethod
def get_macro_defaults():
    """Return default macro scores with data sources."""
    macro_data = {
        "1. Global Macroeconomic Factors (12%)": {
            "weight": 12.0,
            "factors": {
                "1.1 Crude Oil Prices": {
                    "sub_weight": 2.5,
                    "default_score": 5,
                    "description": "Brent/WTI trend, OPEC decisions",
                    "source": (
                        "https://tradingeconomics.com/commodity/crude-oil | "
                        "https://www.moneycontrol.com/commodity/crude-oil-price"
                    ),
                    "how_to_check": (
                        "Check current Brent crude price. "
                        "Below $70=Bullish for India, $70-90=Neutral, Above $90=Bearish"
                    ),
                },
                "1.2 US Dollar Index (DXY)": {
                    "sub_weight": 2.0,
                    "default_score": 5,
                    "description": "Dollar strength/weakness, USD/INR trend",
                    "source": (
                        "https://www.investing.com/currencies/us-dollar-index | "
                        "RBI reference rate"
                    ),
                    "how_to_check": (
                        "DXY below 100=Bullish for EMs, 100-105=Neutral, "
                        "Above 105=Bearish"
                    ),
                },
                "1.3 US Federal Reserve Policy": {
                    "sub_weight": 1.5,
                    "default_score": 5,
                    "description": "Fed funds rate, QE/QT, commentary",
                    "source": (
                        "https://www.federalreserve.gov/ | CME FedWatch Tool"
                    ),
                    "how_to_check": (
                        "Rate cuts expected=Bullish, Pause=Neutral, Hikes=Bearish"
                    ),
                },
                "1.4 Global GDP Growth": {
                    "sub_weight": 1.0,
                    "default_score": 6,
                    "description": "IMF/World Bank projections",
                    "source": (
                        "https://www.imf.org/en/Publications/WEO | "
                        "World Bank Global Economic Prospects"
                    ),
                    "how_to_check": (
                        "Global GDP >3%=Bullish, 2-3%=Neutral, <2%=Bearish"
                    ),
                },
                "1.5 Global Inflation Trends": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "US CPI/PPI, EU inflation",
                    "source": "https://www.bls.gov/cpi/ | Eurostat",
                    "how_to_check": (
                        "Inflation declining toward target=Bullish, "
                        "Sticky=Neutral, Rising=Bearish"
                    ),
                },
                "1.6 Gold & Commodity Prices": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "Gold, silver, copper, aluminum, steel",
                    "source": (
                        "https://www.moneycontrol.com/commodity/ | LME prices"
                    ),
                    "how_to_check": (
                        "Stable/declining=Bullish for consumers, "
                        "Rising=Bullish for producers"
                    ),
                },
                "1.7 Global Bond Yields": {
                    "sub_weight": 0.8,
                    "default_score": 5,
                    "description": "US 10Y yield, yield curve",
                    "source": (
                        "https://www.cnbc.com/quotes/US10Y | "
                        "https://fred.stlouisfed.org/"
                    ),
                    "how_to_check": (
                        "10Y declining=Bullish for equities, Flat=Neutral, "
                        "Rising sharply=Bearish"
                    ),
                },
                "1.8 Geopolitical Risks": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "Wars, trade wars, sanctions",
                    "source": (
                        "Global news - Reuters, Bloomberg | "
                        "Council on Foreign Relations"
                    ),
                    "how_to_check": (
                        "Low tension=Bullish(8), Moderate=Neutral(5), "
                        "High=Bearish(2)"
                    ),
                },
                "1.9 FII/FPI Flows": {
                    "sub_weight": 1.2,
                    "default_score": 5,
                    "description": "Net FII buying/selling in Indian markets",
                    "source": (
                        "https://www.moneycontrol.com/stocks/marketstats/"
                        "fii_dii_activity/index.php | NSDL FPI Monitor"
                    ),
                    "how_to_check": (
                        "Net buyers >5000cr/month=Bullish, Flat=Neutral, "
                        "Net sellers=Bearish"
                    ),
                },
            },
        },
        "2. RBI & Monetary Policy (7%)": {
            "weight": 7.0,
            "factors": {
                "2.1 RBI Repo Rate & Stance": {
                    "sub_weight": 1.5,
                    "default_score": 6,
                    "description": "Current rate, trajectory",
                    "source": (
                        "https://www.rbi.org.in/Scripts/"
                        "BS_PressReleaseDisplay.aspx | "
                        "RBI Monetary Policy Statement"
                    ),
                    "how_to_check": (
                        "Rate cuts/Accommodative=Bullish, "
                        "Neutral stance=Neutral, Hawkish/Hikes=Bearish"
                    ),
                },
                "2.2 RBI CRR/SLR": {
                    "sub_weight": 0.8,
                    "default_score": 5,
                    "description": "Liquidity impact on banking",
                    "source": "RBI website - Current CRR and SLR rates",
                    "how_to_check": (
                        "CRR cut=Bullish(liquidity release), "
                        "Stable=Neutral, Hike=Bearish"
                    ),
                },
                "2.3 RBI Liquidity Management": {
                    "sub_weight": 0.7,
                    "default_score": 5,
                    "description": "OMO, LAF, MSF operations",
                    "source": (
                        "RBI - Money Market Operations | "
                        "RBI Weekly Statistical Supplement"
                    ),
                    "how_to_check": (
                        "Surplus liquidity=Bullish, Balanced=Neutral, "
                        "Tight=Bearish"
                    ),
                },
                "2.4 India CPI/WPI Inflation": {
                    "sub_weight": 1.0,
                    "default_score": 6,
                    "description": "Core, food, fuel inflation",
                    "source": (
                        "https://mospi.gov.in/ | RBI Inflation data | CMIE"
                    ),
                    "how_to_check": (
                        "CPI <4%=Bullish, 4-6%=Neutral, >6%=Bearish"
                    ),
                },
                "2.5 INR Exchange Rate Stability": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "RBI intervention, REER",
                    "source": (
                        "RBI Reference Rate | FBIL | "
                        "https://www.x-rates.com/"
                    ),
                    "how_to_check": (
                        "INR stable/appreciating=Bullish, "
                        "Mild depreciation=Neutral, Sharp fall=Bearish"
                    ),
                },
                "2.6 Credit Growth": {
                    "sub_weight": 0.5,
                    "default_score": 6,
                    "description": "Bank credit growth",
                    "source": (
                        "RBI - Sectoral Deployment of Bank Credit | "
                        "Scheduled Commercial Banks data"
                    ),
                    "how_to_check": (
                        ">15% YoY=Bullish, 10-15%=Neutral, <10%=Bearish"
                    ),
                },
                "2.7 RBI Sector-Specific Norms": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "NPA norms, NBFC regulations",
                    "source": (
                        "RBI Circulars and Notifications | "
                        "RBI Master Directions"
                    ),
                    "how_to_check": (
                        "Relaxation=Bullish, Status quo=Neutral, "
                        "Tightening=Bearish"
                    ),
                },
                "2.8 Money Supply (M3)": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "Broad money growth",
                    "source": (
                        "RBI Weekly Statistical Supplement - Money Supply"
                    ),
                    "how_to_check": (
                        "M3 growth >10%=Bullish, 8-10%=Neutral, "
                        "<8%=Bearish"
                    ),
                },
            },
        },
        "3. Govt & Fiscal Policy (8%)": {
            "weight": 8.0,
            "factors": {
                "3.1 Union Budget & Fiscal Deficit": {
                    "sub_weight": 1.5,
                    "default_score": 6,
                    "description": "Fiscal deficit target, capex allocation",
                    "source": (
                        "https://www.indiabudget.gov.in/ | "
                        "CGA Monthly Accounts"
                    ),
                    "how_to_check": (
                        "Fiscal deficit <5.5% of GDP and "
                        "high capex=Bullish"
                    ),
                },
                "3.2 Tax Policy Changes": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "Corporate tax, capital gains tax, STT",
                    "source": (
                        "Union Budget documents | CBDT notifications"
                    ),
                    "how_to_check": (
                        "Tax cuts=Bullish, No change=Neutral, "
                        "Tax hikes=Bearish"
                    ),
                },
                "3.3 GST Rates & Collections": {
                    "sub_weight": 0.8,
                    "default_score": 6,
                    "description": "Rate changes, collections trend",
                    "source": (
                        "https://pib.gov.in/ - Monthly GST Revenue | "
                        "GST Council press releases"
                    ),
                    "how_to_check": (
                        "Collections >1.8L Cr/month and "
                        "stable rates=Bullish"
                    ),
                },
                "3.4 PLI Schemes & Subsidies": {
                    "sub_weight": 1.0,
                    "default_score": 7,
                    "description": (
                        "Sector-specific PLI, FAME, semiconductor"
                    ),
                    "source": (
                        "DPIIT - PLI Scheme details | "
                        "Ministry of Commerce"
                    ),
                    "how_to_check": (
                        "Active PLI for relevant sector=Bullish, "
                        "Not applicable=Neutral"
                    ),
                },
                "3.5 Infrastructure Spending": {
                    "sub_weight": 0.8,
                    "default_score": 7,
                    "description": "NIP, PM Gati Shakti, capex",
                    "source": (
                        "Budget Capex data | NHAI, NHIDCL, "
                        "Railways CapEx data"
                    ),
                    "how_to_check": (
                        "Capex >10L Cr/year growing=Bullish"
                    ),
                },
                "3.6 Import/Export Duties & Trade Policy": {
                    "sub_weight": 0.7,
                    "default_score": 5,
                    "description": "Custom duties, anti-dumping, FTAs",
                    "source": (
                        "CBIC notifications | DGTR | Commerce Ministry"
                    ),
                    "how_to_check": (
                        "Favorable duty structure for sector=Bullish"
                    ),
                },
                "3.7 Disinvestment & Privatization": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "PSU stake sales",
                    "source": (
                        "DIPAM website | Budget disinvestment target"
                    ),
                    "how_to_check": (
                        "Active disinvestment=Bullish for PSUs, "
                        "Stalled=Neutral"
                    ),
                },
                "3.8 Political Stability": {
                    "sub_weight": 0.7,
                    "default_score": 7,
                    "description": "Central govt stability, elections",
                    "source": (
                        "News sources | "
                        "ECI - Election Commission schedule"
                    ),
                    "how_to_check": (
                        "Stable majority govt, "
                        "no imminent elections=Bullish(8)"
                    ),
                },
                "3.9 India GDP Growth": {
                    "sub_weight": 1.0,
                    "default_score": 7,
                    "description": "Real GDP, GVA trends",
                    "source": (
                        "https://mospi.gov.in/ | "
                        "CSO Advance Estimates | RBI projections"
                    ),
                    "how_to_check": (
                        "GDP >7%=Bullish, 5-7%=Neutral, <5%=Bearish"
                    ),
                },
            },
        },
        "4. Regulatory & Legal Environment (5%)": {
            "weight": 5.0,
            "factors": {
                "4.1 SEBI Regulations": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "Disclosure norms, margin rules",
                    "source": (
                        "https://www.sebi.gov.in/sebiweb/home/"
                        "HomeAction.do?doListing=yes&sid=1&ssid=1"
                    ),
                    "how_to_check": (
                        "Investor-friendly reforms=Bullish, "
                        "Restrictive=Bearish"
                    ),
                },
                "4.2 Sector-Specific Regulators": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": (
                        "TRAI, IRDAI, PFRDA, FSSAI, CDSCO, CERC/SERC"
                    ),
                    "source": "Respective regulator websites",
                    "how_to_check": (
                        "Favorable regulations=Bullish, "
                        "Adverse=Bearish"
                    ),
                },
                "4.3 ESG Regulations": {
                    "sub_weight": 0.8,
                    "default_score": 5,
                    "description": "BRSR mandates, carbon tax",
                    "source": (
                        "SEBI BRSR framework | MoEFCC notifications"
                    ),
                    "how_to_check": (
                        "Company compliant and ahead of curve=Bullish"
                    ),
                },
                "4.4 Labour Law Reforms": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "New labour codes",
                    "source": (
                        "Ministry of Labour | "
                        "4 Labour Codes implementation status"
                    ),
                    "how_to_check": (
                        "Simplified codes implemented="
                        "Bullish for manufacturing"
                    ),
                },
                "4.5 Land Acquisition & Clearances": {
                    "sub_weight": 0.7,
                    "default_score": 5,
                    "description": "Environmental/forest clearance",
                    "source": (
                        "Parivesh portal | MoEFCC | "
                        "State-level portals"
                    ),
                    "how_to_check": (
                        "Faster clearances=Bullish for "
                        "infra/real estate"
                    ),
                },
                "4.6 FDI Policy": {
                    "sub_weight": 0.5,
                    "default_score": 6,
                    "description": "Sector FDI caps, route",
                    "source": (
                        "DPIIT FDI Policy Document | RBI FDI data"
                    ),
                    "how_to_check": (
                        "Liberal FDI in sector=Bullish"
                    ),
                },
                "4.7 Data Privacy & Digital Regulations": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "DPDP Act, data localization",
                    "source": "MeitY | DPDP Act 2023 text",
                    "how_to_check": (
                        "Clear framework=Neutral, "
                        "Ambiguous/restrictive=Bearish for tech"
                    ),
                },
            },
        },
        "5. Market Sentiment & Technicals (4%)": {
            "weight": 4.0,
            "factors": {
                "5.1 DII Flows": {
                    "sub_weight": 0.7,
                    "default_score": 6,
                    "description": "Mutual fund flows, insurance buying",
                    "source": (
                        "AMFI monthly data | "
                        "Moneycontrol FII/DII page"
                    ),
                    "how_to_check": (
                        "Strong MF SIP flows + "
                        "net DII buying=Bullish"
                    ),
                },
                "5.2 Market Valuation (Nifty PE/PB)": {
                    "sub_weight": 0.8,
                    "default_score": 5,
                    "description": "Nifty PE vs historical average",
                    "source": (
                        "https://www.niftyindices.com/reports/"
                        "historical-data | NSE website"
                    ),
                    "how_to_check": (
                        "Nifty PE <20=Bullish, 20-24=Neutral, "
                        ">24=Expensive"
                    ),
                },
                "5.3 India VIX": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "Fear gauge",
                    "source": (
                        "NSE India VIX | https://www.nseindia.com/"
                    ),
                    "how_to_check": (
                        "VIX <13=Low fear(Bullish), "
                        "13-20=Neutral, >20=High fear"
                    ),
                },
                "5.4 Sector Rotation Trends": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "Inflows/outflows by sector",
                    "source": (
                        "NSE Sectoral Indices performance | "
                        "MF sectoral allocation data"
                    ),
                    "how_to_check": (
                        "Money flowing into stock sector=Bullish"
                    ),
                },
                "5.5 IPO & Primary Market": {
                    "sub_weight": 0.3,
                    "default_score": 5,
                    "description": "IPO pipeline, listing performance",
                    "source": (
                        "https://www.chittorgarh.com/ipo/ | "
                        "SEBI IPO tracker"
                    ),
                    "how_to_check": (
                        "Healthy IPO market=Bullish sentiment, "
                        "Excessive=Frothy"
                    ),
                },
                "5.6 Global Contagion Risk": {
                    "sub_weight": 0.7,
                    "default_score": 5,
                    "description": "Correlation with global markets",
                    "source": (
                        "S&P500, Hang Seng, Nikkei correlation | "
                        "Bloomberg"
                    ),
                    "how_to_check": (
                        "Global markets stable=Bullish, "
                        "Volatile=Bearish"
                    ),
                },
                "5.7 Retail Participation": {
                    "sub_weight": 0.5,
                    "default_score": 6,
                    "description": "Demat growth, retail AUM",
                    "source": (
                        "CDSL/NSDL demat data | SEBI Annual Report"
                    ),
                    "how_to_check": (
                        "Growing demat accounts=Long-term Bullish"
                    ),
                },
            },
        },
        "6. Structural & Demographic (4%)": {
            "weight": 4.0,
            "factors": {
                "6.1 India Demographics": {
                    "sub_weight": 0.7,
                    "default_score": 7,
                    "description": "Working age population, urbanization",
                    "source": (
                        "Census data | UN Population Projections | "
                        "World Bank"
                    ),
                    "how_to_check": (
                        "Young population, rising urbanization="
                        "Structural Bullish"
                    ),
                },
                "6.2 Digital Penetration": {
                    "sub_weight": 0.5,
                    "default_score": 8,
                    "description": "UPI volumes, internet users",
                    "source": (
                        "NPCI UPI data | TRAI subscriber data | "
                        "Statista"
                    ),
                    "how_to_check": (
                        "Rising digital adoption="
                        "Bullish for tech/fintech"
                    ),
                },
                "6.3 Monsoon & Agriculture": {
                    "sub_weight": 1.0,
                    "default_score": 5,
                    "description": "IMD forecast, El Nino/La Nina",
                    "source": (
                        "https://mausam.imd.gov.in/ | "
                        "IMD Monsoon forecast"
                    ),
                    "how_to_check": (
                        "Normal monsoon=Bullish for rural/FMCG, "
                        "Deficient=Bearish"
                    ),
                },
                "6.4 Real Estate Cycle": {
                    "sub_weight": 0.5,
                    "default_score": 6,
                    "description": "Housing sales, inventory",
                    "source": (
                        "PropEquity | Knight Frank India | "
                        "JLL India Reports"
                    ),
                    "how_to_check": (
                        "Rising sales + falling inventory="
                        "Bullish for RE/cement/home finance"
                    ),
                },
                "6.5 Employment": {
                    "sub_weight": 0.5,
                    "default_score": 5,
                    "description": "CMIE data, EPFO additions",
                    "source": (
                        "CMIE unemployment data | "
                        "EPFO payroll data | PLFS survey"
                    ),
                    "how_to_check": (
                        "Falling unemployment, "
                        "rising EPFO additions=Bullish"
                    ),
                },
                "6.6 Consumer Confidence": {
                    "sub_weight": 0.3,
                    "default_score": 5,
                    "description": "RBI consumer confidence survey",
                    "source": (
                        "RBI Consumer Confidence Survey (bimonthly)"
                    ),
                    "how_to_check": (
                        "Improving confidence="
                        "Bullish for consumption"
                    ),
                },
                "6.7 PMI (Mfg & Services)": {
                    "sub_weight": 0.5,
                    "default_score": 6,
                    "description": "Expansion/contraction",
                    "source": (
                        "S&P Global India PMI | "
                        "https://www.pmi.spglobal.com/"
                    ),
                    "how_to_check": (
                        "PMI >52=Bullish, 50-52=Neutral, "
                        "<50=Bearish"
                    ),
                },
            },
        },
    }
    return macro_data

# =============================================================================
# MICRO ANALYZER
# =============================================================================

class MicroAnalyzer:
"""Analyzes micro/stock-specific factors using fetched data."""

def __init__(self, fetcher, sector):
    self.f = fetcher
    self.info = fetcher.info
    self.sector = sector

def safe_get(self, key, default=None):
    """Safely get a value from info dict."""
    return self.info.get(key, default)

def analyze_financials(self):
    """Section 7: Financial Performance (15%)."""
    results = {}

    # 7.1 Revenue Growth
    rev_growth = self.safe_get("revenueGrowth")
    rev_score = 5
    if rev_growth is not None:
        if rev_growth > 0.20:
            rev_score = 9
        elif rev_growth > 0.10:
            rev_score = 7
        elif rev_growth > 0.05:
            rev_score = 5
        elif rev_growth > 0:
            rev_score = 4
        else:
            rev_score = 2
    rev_value = "N/A"
    if rev_growth is not None:
        rev_value = "{:.1f}%".format(rev_growth * 100)
    results["7.1 Revenue Growth"] = {
        "sub_weight": 2.5,
        "value": rev_value,
        "score": rev_score,
        "source": (
            "Yahoo Finance - " + self.f.yahoo_ticker +
            " revenueGrowth field"
        ),
        "how_to_check": (
            "Screener.in > Company > P&L > Revenue CAGR; "
            "Tijori Finance; MoneyControl Financials"
        ),
    }

    # 7.2 Profitability (EBITDA / PAT Margin)
    ebitda_margin = self.safe_get("ebitdaMargins")
    profit_margin = self.safe_get("profitMargins")
    prof_score = 5
    if ebitda_margin is not None:
        if ebitda_margin > 0.25:
            prof_score = 9
        elif ebitda_margin > 0.15:
            prof_score = 7
        elif ebitda_margin > 0.10:
            prof_score = 5
        else:
            prof_score = 3
    prof_value = "N/A"
    if ebitda_margin is not None and profit_margin is not None:
        prof_value = "EBITDA: {:.1f}%, PAT: {:.1f}%".format(
            ebitda_margin * 100, profit_margin * 100
        )
    results["7.2 Profitability (EBITDA/PAT Margin)"] = {
        "sub_weight": 2.5,
        "value": prof_value,
        "score": prof_score,
        "source": "Yahoo Finance - ebitdaMargins, profitMargins",
        "how_to_check": (
            "Screener.in > Company > Ratios; "
            "Tijori Finance > Profitability"
        ),
    }

    # 7.3 EPS Growth
    earnings_growth = self.safe_get("earningsGrowth")
    eps_score = 5
    if earnings_growth is not None:
        if earnings_growth > 0.25:
            eps_score = 9
        elif earnings_growth > 0.15:
            eps_score = 7
        elif earnings_growth > 0.05:
            eps_score = 5
        elif earnings_growth > 0:
            eps_score = 4
        else:
            eps_score = 2
    eps_value = "N/A"
    if earnings_growth is not None:
        eps_value = "{:.1f}%".format(earnings_growth * 100)
    results["7.3 EPS Growth"] = {
        "sub_weight": 1.5,
        "value": eps_value,
        "score": eps_score,
        "source": "Yahoo Finance - earningsGrowth",
        "how_to_check": (
            "Screener.in > Company > EPS trend; "
            "Tickertape > Financials"
        ),
    }

    # 7.4 Return Ratios (ROE / ROA)
    roe = self.safe_get("returnOnEquity")
    roa = self.safe_get("returnOnAssets")
    ret_score = 5
    if roe is not None:
        if roe > 0.20:
            ret_score = 9
        elif roe > 0.15:
            ret_score = 7
        elif roe > 0.10:
            ret_score = 5
        else:
            ret_score = 3
    ret_value = "N/A"
    if roe is not None and roa is not None:
        ret_value = "ROE: {:.1f}%, ROA: {:.1f}%".format(
            roe * 100, roa * 100
        )
    results["7.4 Return Ratios (ROE/ROA)"] = {
        "sub_weight": 2.0,
        "value": ret_value,
        "score": ret_score,
        "source": (
            "Yahoo Finance - returnOnEquity, returnOnAssets"
        ),
        "how_to_check": (
            "Screener.in > Company > Ratios tab; "
            "ROCE from Tijori Finance"
        ),
    }

    # 7.5 Cash Flow Analysis
    ocf = self.safe_get("operatingCashflow")
    fcf = self.safe_get("freeCashflow")
    cf_score = 5
    if fcf is not None and ocf is not None:
        if fcf > 0 and ocf > 0:
            cf_score = 7
            market_cap = self.safe_get("marketCap", 1)
            if market_cap and market_cap > 0:
                fcf_yield = fcf / market_cap
            else:
                fcf_yield = 0
            if fcf_yield > 0.05:
                cf_score = 9
            elif fcf_yield > 0.03:
                cf_score = 7
        elif ocf is not None and ocf > 0:
            cf_score = 5
        else:
            cf_score = 2
    cf_value = "N/A"
    if ocf is not None and fcf is not None:
        cf_value = "OCF: Rs{:.0f}Cr, FCF: Rs{:.0f}Cr".format(
            ocf / 1e7, fcf / 1e7
        )
    results["7.5 Cash Flow Analysis"] = {
        "sub_weight": 2.0,
        "value": cf_value,
        "score": cf_score,
        "source": (
            "Yahoo Finance - operatingCashflow, freeCashflow"
        ),
        "how_to_check": (
            "Screener.in > Cash Flow; "
            "Tijori Finance > Cash Flow"
        ),
    }

    # 7.6 Working Capital Efficiency
    results["7.6 Working Capital Efficiency"] = {
        "sub_weight": 1.0,
        "value": "Refer detailed source",
        "score": 5,
        "source": "Manual check needed",
        "how_to_check": (
            "Screener.in > Balance Sheet > Debtor Days, "
            "Inventory Days; Tijori Finance > Efficiency Ratios"
        ),
    }

    # 7.7 Balance Sheet Strength
    de_ratio = self.safe_get("debtToEquity")
    current_ratio = self.safe_get("currentRatio")
    bs_score = 5
    if de_ratio is not None:
        if de_ratio < 0.3:
            bs_score = 9
        elif de_ratio < 0.7:
            bs_score = 7
        elif de_ratio < 1.5:
            bs_score = 5
        else:
            bs_score = 3
    bs_value = "N/A"
    if de_ratio is not None and current_ratio is not None:
        bs_value = "D/E: {:.2f}, Current Ratio: {:.2f}".format(
            de_ratio, current_ratio
        )
    results["7.7 Balance Sheet Strength"] = {
        "sub_weight": 2.0,
        "value": bs_value,
        "score": bs_score,
        "source": (
            "Yahoo Finance - debtToEquity, currentRatio"
        ),
        "how_to_check": (
            "Screener.in > Balance Sheet; "
            "Moneycontrol > Financials > Ratios"
        ),
    }

    # 7.8 Dividend History
    div_yield = self.safe_get("dividendYield")
    div_score = 5
    if div_yield is not None:
        if div_yield > 0.03:
            div_score = 8
        elif div_yield > 0.01:
            div_score = 6
        elif div_yield > 0:
            div_score = 4
        else:
            div_score = 3
    div_value = "N/A"
    if div_yield is not None:
        div_value = "{:.2f}%".format(div_yield * 100)
    results["7.8 Dividend History"] = {
        "sub_weight": 0.5,
        "value": div_value,
        "score": div_score,
        "source": "Yahoo Finance - dividendYield",
        "how_to_check": (
            "Screener.in > Company > Dividend History; "
            "Tickertape > Dividends"
        ),
    }

    # 7.9 Earnings Quality
    results["7.9 Earnings Quality"] = {
        "sub_weight": 1.0,
        "value": "Refer source for audit report and accrual analysis",
        "score": 5,
        "source": "Annual Report - Auditor Report section",
        "how_to_check": (
            "Check Annual Report for audit qualifications; "
            "Tofler.in for filings; Screener.in > Docs"
        ),
    }

    return results

def analyze_valuation(self):
    """Section 8: Valuation (8%)."""
    results = {}

    # 8.1 PE Ratio
    pe_trailing = self.safe_get("trailingPE")
    pe_forward = self.safe_get("forwardPE")
    peg = self.safe_get("pegRatio")
    pe_score = 5
    if pe_forward is not None and pe_trailing is not None:
        if pe_forward < 15:
            pe_score = 8
        elif pe_forward < 25:
            pe_score = 6
        elif pe_forward < 40:
            pe_score = 4
        else:
            pe_score = 2
        if pe_forward < pe_trailing:
            pe_score = min(10, pe_score + 1)

    pe_value = "N/A"
    if pe_trailing is not None and pe_forward is not None and peg is not None:
        pe_value = "TTM: {:.1f}, Fwd: {:.1f}, PEG: {:.2f}".format(
            pe_trailing, pe_forward, peg
        )
    elif pe_trailing is not None:
        pe_value = "TTM: {:.1f}".format(pe_trailing)

    results["8.1 PE Ratio (TTM & Forward)"] = {
        "sub_weight": 1.5,
        "value": pe_value,
        "score": pe_score,
        "source": (
            "Yahoo Finance - trailingPE, forwardPE, pegRatio"
        ),
        "how_to_check": (
            "Screener.in > Company > PE chart; "
            "Tickertape > Valuation; "
            "Compare with sector PE on Tijori"
        ),
    }

    # 8.2 EV/EBITDA
    ev_ebitda = self.safe_get("enterpriseToEbitda")
    ev_score = 5
    if ev_ebitda is not None:
        if ev_ebitda < 10:
            ev_score = 8
        elif ev_ebitda < 18:
            ev_score = 6
        elif ev_ebitda < 30:
            ev_score = 4
        else:
            ev_score = 2
    ev_value = "N/A"
    if ev_ebitda is not None:
        ev_value = "{:.1f}".format(ev_ebitda)
    results["8.2 EV/EBITDA"] = {
        "sub_weight": 1.5,
        "value": ev_value,
        "score": ev_score,
        "source": "Yahoo Finance - enterpriseToEbitda",
        "how_to_check": (
            "Screener.in > Ratios; Tickertape > EV/EBITDA; "
            "Compare peers on Tijori Finance"
        ),
    }

    # 8.3 Price/Book
    pb = self.safe_get("priceToBook")
    pb_score = 5
    if pb is not None:
        if self.sector in ["banking", "realestate"]:
            if pb < 1.5:
                pb_score = 8
            elif pb < 3.0:
                pb_score = 6
            else:
                pb_score = 3
        else:
            if pb < 3:
                pb_score = 7
            elif pb < 6:
                pb_score = 5
            else:
                pb_score = 3
    pb_value = "N/A"
    if pb is not None:
        pb_value = "{:.2f}".format(pb)
    results["8.3 Price/Book Value"] = {
        "sub_weight": 1.0,
        "value": pb_value,
        "score": pb_score,
        "source": "Yahoo Finance - priceToBook",
        "how_to_check": (
            "Screener.in > Ratios; "
            "NSE website > Company > Key Ratios"
        ),
    }

    # 8.4 Price/Sales
    ps = self.safe_get("priceToSalesTrailing12Months")
    ps_score = 5
    if ps is not None:
        if ps < 2:
            ps_score = 8
        elif ps < 5:
            ps_score = 6
        elif ps < 10:
            ps_score = 4
        else:
            ps_score = 2
    ps_value = "N/A"
    if ps is not None:
        ps_value = "{:.2f}".format(ps)
    results["8.4 Price/Sales"] = {
        "sub_weight": 0.5,
        "value": ps_value,
        "score": ps_score,
        "source": (
            "Yahoo Finance - priceToSalesTrailing12Months"
        ),
        "how_to_check": "Tickertape > Valuation; Screener.in",
    }

    # 8.5 DCF / Intrinsic Value
    results["8.5 DCF / Intrinsic Value"] = {
        "sub_weight": 2.0,
        "value": "Requires manual DCF model",
        "score": 5,
        "source": (
            "Build DCF using Screener.in financials or use "
            "Tickertape/Trendlyne intrinsic value"
        ),
        "how_to_check": (
            "Tickertape > Fair Value; "
            "Trendlyne > Forecaster; "
            "Simply Wall St > Valuation"
        ),
    }

    # 8.6 Margin of Safety
    target_price = self.safe_get("targetMeanPrice")
    current_price = self.safe_get("currentPrice")
    mos_score = 5
    mos_val = "N/A"
    if (target_price is not None and current_price is not None
            and current_price > 0):
        upside = (target_price - current_price) / current_price
        mos_val = (
            "Target: Rs{:.0f}, CMP: Rs{:.0f}, "
            "Upside: {:.1f}%"
        ).format(target_price, current_price, upside * 100)
        if upside > 0.30:
            mos_score = 9
        elif upside > 0.15:
            mos_score = 7
        elif upside > 0:
            mos_score = 5
        else:
            mos_score = 3
    results["8.6 Margin of Safety"] = {
        "sub_weight": 1.0,
        "value": mos_val,
        "score": mos_score,
        "source": (
            "Yahoo Finance - targetMeanPrice (analyst consensus)"
        ),
        "how_to_check": (
            "Trendlyne > Forecaster; "
            "Tickertape > Analyst Estimates; "
            "MoneyControl > Analyst Reco"
        ),
    }

    # 8.7 Relative Valuation
    results["8.7 Relative Valuation"] = {
        "sub_weight": 0.5,
        "value": "Compare PE/EV with sector peers",
        "score": 5,
        "source": (
            "Screener.in > Peers comparison; "
            "Tijori Finance > Industry comparison"
        ),
        "how_to_check": (
            "Screener.in > Company > Peer Comparison tab"
        ),
    }

    return results

def analyze_promoter_management(self):
    """Section 9: Promoter & Management Quality (10%)."""
    results = {}

    # 9.1 Promoter Holding
    held_pct_insiders = self.safe_get("heldPercentInsiders")
    promoter_score = 5
    if held_pct_insiders is not None:
        if held_pct_insiders > 0.60:
            promoter_score = 8
        elif held_pct_insiders > 0.45:
            promoter_score = 7
        elif held_pct_insiders > 0.30:
            promoter_score = 5
        else:
            promoter_score = 3
    prom_value = "N/A"
    if held_pct_insiders is not None:
        prom_value = "{:.1f}%".format(held_pct_insiders * 100)
    results["9.1 Promoter Holding & Trend"] = {
        "sub_weight": 1.5,
        "value": prom_value,
        "score": promoter_score,
        "source": (
            "Yahoo Finance - heldPercentInsiders "
            "(proxy for promoter holding)"
        ),
        "how_to_check": (
            "BSE > Company > Shareholding Pattern; "
            "Trendlyne > Shareholding; "
            "MoneyControl > Shareholding"
        ),
    }

    # 9.2 to 9.10 - Qualitative factors
    qual_factors = [
        (
            "9.2 Promoter Pledge %", 1.0,
            "Check BSE/NSE shareholding pattern",
            "BSE/NSE Shareholding Pattern filings",
            "BSE > Corp Filing > Shareholding Pattern > "
            "Shares Pledged column; Trendlyne > Pledge data"
        ),
        (
            "9.3 Promoter Background & Track Record", 1.5,
            "Qualitative - check source",
            "Company Annual Report > Directors Profile; "
            "LinkedIn; Business news",
            "Annual Report > Corporate Governance section; "
            "Wikipedia; Company website > Leadership"
        ),
        (
            "9.4 Family/Succession Structure", 0.5,
            "Qualitative",
            "Annual Report; Business news archives",
            "Check if professional management or family-run; "
            "any succession plan disclosed"
        ),
        (
            "9.5 Management Compensation", 0.5,
            "Check Annual Report",
            "Annual Report > Corporate Governance > "
            "Remuneration of Directors",
            "Annual Report; Tofler.in > Director Remuneration; "
            "Check if aligned with performance"
        ),
        (
            "9.6 Capital Allocation History", 1.5,
            "Check acquisitions, capex returns",
            "Annual Report > MD&A; Past acquisition "
            "announcements; Screener.in > Capex history",
            "Compare ROCE vs WACC over time; "
            "Check if acquisitions created value"
        ),
        (
            "9.7 Corporate Governance Score", 1.5,
            "Check governance reports",
            "IiAS governance scores; "
            "Annual Report > Corp Gov report",
            "IiAS ratings; Annual Report > Corporate "
            "Governance Report; Check board independence %"
        ),
        (
            "9.8 Insider Trading Activity", 0.5,
            "Check SAST disclosures",
            "BSE > Corp Announcements > SAST; "
            "NSE > Corporate Actions",
            "NSE/BSE > Insider Trading disclosures; "
            "Trendlyne > Bulk/Block deals"
        ),
        (
            "9.9 Promoter Entity Structure", 1.0,
            "Check holding company structure",
            "Annual Report > Related Party Transactions; "
            "MCA (Zauba Corp) for group entities",
            "Zauba Corp / Tofler.in for entity mapping; "
            "Annual Report > Subsidiaries list"
        ),
        (
            "9.10 Management Guidance Accuracy", 0.5,
            "Compare past guidance vs actuals",
            "Earnings call transcripts on Trendlyne / "
            "Screener.in; Past analyst reports",
            "Read last 4 earnings call transcripts; "
            "Compare management guidance with actual numbers"
        ),
    ]

    for name, weight, value, source, how in qual_factors:
        results[name] = {
            "sub_weight": weight,
            "value": value,
            "score": 5,
            "source": source,
            "how_to_check": how,
        }

    return results

def analyze_legal(self):
    """Section 10: Legal & Litigation (5%)."""
    results = {}

    factor_list = [
        (
            "10.1 Open Litigation (Civil)", 0.8,
            "Annual Report > Contingent Liabilities; "
            "Company Legal Disclosures"
        ),
        (
            "10.2 Open Litigation (Criminal)", 0.8,
            "DRHP/Annual Report > Outstanding Litigation; "
            "MCA filings"
        ),
        (
            "10.3 Tax Disputes", 0.7,
            "Annual Report > Notes to Accounts > "
            "Contingent Liabilities"
        ),
        (
            "10.4 Regulatory Penalties", 0.5,
            "SEBI/RBI/Sectoral regulator orders; "
            "Company announcements"
        ),
        (
            "10.5 Environmental/NGT Cases", 0.5,
            "NGT website; MoEFCC; Company ESG report"
        ),
        (
            "10.6 IP/Patent Disputes", 0.4,
            "Relevant for pharma/tech; Company Annual Report"
        ),
        (
            "10.7 Labour Disputes", 0.3,
            "Annual Report; Labour court filings; News"
        ),
        (
            "10.8 Closed Litigation Outcomes", 0.5,
            "Annual Report > Contingent Liabilities; "
            "Legal databases"
        ),
        (
            "10.9 Whistleblower/Fraud Allegations", 0.5,
            "News search; Short-seller reports; SEBI orders"
        ),
    ]

    for name, w, src in factor_list:
        results[name] = {
            "sub_weight": w,
            "value": (
                "Check source - qualitative assessment needed"
            ),
            "score": 5,
            "source": src,
            "how_to_check": (
                "Annual Report > Notes to Accounts; "
                "BSE/NSE Announcements; "
                "Google News search for "
                "'" + self.f.symbol + " litigation/case'"
            ),
        }

    return results

def analyze_competitive(self):
    """Section 11: Competitive Positioning (7%)."""
    results = {}
    industry = self.safe_get("industry", "N/A")
    sector_name = self.safe_get("sector", "N/A")

    results["11.1 Market Share & Trend"] = {
        "sub_weight": 1.5,
        "value": "Industry: " + str(industry),
        "score": 5,
        "source": (
            "Industry reports (CRISIL, ICRA, CARE); "
            "Company investor presentations"
        ),
        "how_to_check": (
            "Google '[Company] market share India'; "
            "Annual Report > MD&A; "
            "Investor presentations on BSE/Company website"
        ),
    }

    results["11.2 Entry Barriers / Moat"] = {
        "sub_weight": 1.5,
        "value": "Qualitative assessment needed",
        "score": 5,
        "source": (
            "Industry analysis; Company Annual Report"
        ),
        "how_to_check": (
            "Evaluate: Brand strength, patents, licenses, "
            "network effects, switching costs, scale advantages"
        ),
    }

    results["11.3 Competitive Intensity"] = {
        "sub_weight": 1.0,
        "value": "Sector: " + str(sector_name),
        "score": 5,
        "source": (
            "Industry reports; "
            "Number of listed peers on Screener.in"
        ),
        "how_to_check": (
            "Screener.in > Peers; "
            "Check number of players, pricing trends"
        ),
    }

    results["11.4 Product Differentiation"] = {
        "sub_weight": 0.5,
        "value": "Check company product portfolio",
        "score": 5,
        "source": "Company website; Investor presentation",
        "how_to_check": (
            "Is product commodity or differentiated? "
            "Check brand premium, product mix"
        ),
    }

    results["11.5 Customer Concentration"] = {
        "sub_weight": 0.5,
        "value": "Check Annual Report",
        "score": 5,
        "source": (
            "Annual Report > Segment reporting; "
            "Revenue breakup in investor presentation"
        ),
        "how_to_check": (
            "Annual Report > Notes > Related party / Segment; "
            "If top 5 clients >50% revenue = risky"
        ),
    }

    results["11.6 Supplier Concentration"] = {
        "sub_weight": 0.5,
        "value": "Check RM sourcing details",
        "score": 5,
        "source": (
            "Annual Report > MD&A; Investor presentation"
        ),
        "how_to_check": (
            "Single-source RM dependency = risky; "
            "Diversified sourcing = safer"
        ),
    }

    results["11.7 Threat of Disruption"] = {
        "sub_weight": 1.0,
        "value": (
            "Evaluate technology and regulatory disruption risk"
        ),
        "score": 5,
        "source": (
            "Industry analysis; News; "
            "Global trends in the sector"
        ),
        "how_to_check": (
            "Is the business model at risk from technology "
            "change? EV vs ICE, fintech vs banks, etc."
        ),
    }

    results["11.8 Geographic Diversification"] = {
        "sub_weight": 0.5,
        "value": "Check export revenue %",
        "score": 5,
        "source": (
            "Annual Report > Segment/Geographic revenue; "
            "Screener.in > Segments"
        ),
        "how_to_check": (
            "Screener.in > Segments tab; "
            "Higher export % = diversified but FX risk"
        ),
    }

    return results

def analyze_growth(self):
    """Section 12: Growth Drivers & Strategy (7%)."""
    results = {}
    rev_growth = self.safe_get("revenueGrowth")
    earnings_growth = self.safe_get("earningsGrowth")

    results["12.1 Order Book / Revenue Pipeline"] = {
        "sub_weight": 1.5,
        "value": (
            "Check investor presentation for order book"
        ),
        "score": 5,
        "source": (
            "Quarterly investor presentations; "
            "BSE announcements; Earnings call transcripts"
        ),
        "how_to_check": (
            "Company investor presentation > "
            "Order Book / Pipeline slide; "
            "Trendlyne > Earnings Call"
        ),
    }

    results["12.2 Capacity Expansion Plans"] = {
        "sub_weight": 1.0,
        "value": "Check capex plans",
        "score": 5,
        "source": (
            "Annual Report > MD&A; "
            "Investor presentation; "
            "Earnings call transcripts"
        ),
        "how_to_check": (
            "Screener.in > Cash Flow > Capex; "
            "Compare capex to depreciation; "
            "Check capacity utilization"
        ),
    }

    results["12.3 R&D / Innovation Spend"] = {
        "sub_weight": 0.5,
        "value": "Check Annual Report",
        "score": 5,
        "source": (
            "Annual Report > R&D expenditure note; "
            "Patent filings"
        ),
        "how_to_check": (
            "Annual Report > Notes > R&D spend; "
            "Google Patents for company filings"
        ),
    }

    results["12.4 Acquisition Strategy"] = {
        "sub_weight": 0.5,
        "value": "Check recent M&A activity",
        "score": 5,
        "source": (
            "BSE/NSE Announcements; News; "
            "VCCEdge for deal data"
        ),
        "how_to_check": (
            "Check last 3-5 year acquisition history; "
            "Were they value-accretive?"
        ),
    }

    results["12.5 New Market / Segment Entry"] = {
        "sub_weight": 0.5,
        "value": "Check diversification plans",
        "score": 5,
        "source": (
            "Investor presentation; "
            "Annual Report > Strategy section"
        ),
        "how_to_check": (
            "Is the company entering new geographies "
            "or product segments?"
        ),
    }

    results["12.6 Digital Transformation"] = {
        "sub_weight": 0.5,
        "value": "Check tech adoption",
        "score": 5,
        "source": (
            "Annual Report > Technology section; "
            "Investor presentation"
        ),
        "how_to_check": (
            "Is the company investing in automation, "
            "AI, digital channels?"
        ),
    }

    # 12.7 Management Guidance
    guidance_score = 5
    guidance_value = "Check earnings call"
    if rev_growth is not None and earnings_growth is not None:
        guidance_value = (
            "Rev Growth: {:.1f}%, Earnings Growth: {:.1f}%"
        ).format(rev_growth * 100, earnings_growth * 100)
        if rev_growth > 0.1:
            guidance_score = 6
    results["12.7 Management Guidance"] = {
        "sub_weight": 1.0,
        "value": guidance_value,
        "score": guidance_score,
        "source": (
            "Latest earnings call transcript; Analyst reports"
        ),
        "how_to_check": (
            "Trendlyne > Earnings Call Transcripts; "
            "Screener.in > Documents > Con Calls"
        ),
    }

    results["12.8 Industry Tailwinds"] = {
        "sub_weight": 1.0,
        "value": "Sector-specific assessment",
        "score": 5,
        "source": (
            "IBEF sector reports; "
            "CRISIL/ICRA industry outlook; "
            "PLI scheme details"
        ),
        "how_to_check": (
            "Google 'India [sector] outlook 2026'; "
            "Check IBEF.org sector page"
        ),
    }

    results["12.9 ESG Initiatives"] = {
        "sub_weight": 0.5,
        "value": "Check BRSR / ESG report",
        "score": 5,
        "source": (
            "Company BRSR Report (mandatory for top 1000); "
            "ESG ratings from Sustainalytics/MSCI"
        ),
        "how_to_check": (
            "Annual Report > BRSR section; "
            "Sustainalytics.com; CRISIL ESG scores"
        ),
    }

    return results

def analyze_shareholding(self):
    """Section 13: Shareholding & Institutional Interest (4%)."""
    results = {}

    fii_pct = self.safe_get("heldPercentInstitutions")
    fii_score = 5
    if fii_pct is not None:
        if fii_pct > 0.40:
            fii_score = 8
        elif fii_pct > 0.25:
            fii_score = 7
        elif fii_pct > 0.10:
            fii_score = 5
        else:
            fii_score = 3
    fii_value = "N/A"
    if fii_pct is not None:
        fii_value = "Institutional: {:.1f}%".format(fii_pct * 100)
    results["13.1 FII Holding & Trend"] = {
        "sub_weight": 0.8,
        "value": fii_value,
        "score": fii_score,
        "source": (
            "Yahoo Finance - heldPercentInstitutions"
        ),
        "how_to_check": (
            "BSE > Shareholding Pattern > FII/FPI %; "
            "Trendlyne > Shareholding > FII trend quarterly"
        ),
    }

    results["13.2 DII/MF Holding & Trend"] = {
        "sub_weight": 0.8,
        "value": "Check BSE shareholding pattern",
        "score": 5,
        "source": (
            "BSE Shareholding Pattern; "
            "AMFI > MF Portfolio disclosure"
        ),
        "how_to_check": (
            "Trendlyne > MF Holding; "
            "MoneyControl > MF tab; AMFI monthly disclosure"
        ),
    }

    results["13.3 Retail vs Institutional Mix"] = {
        "sub_weight": 0.3,
        "value": "Check shareholding pattern",
        "score": 5,
        "source": "BSE Shareholding Pattern",
        "how_to_check": (
            "High retail (>30%) = more volatile; "
            "Institutional-heavy = more stable"
        ),
    }

    results["13.4 Bulk/Block Deals"] = {
        "sub_weight": 0.3,
        "value": "Check recent deals",
        "score": 5,
        "source": (
            "NSE > Market Data > Bulk Deals; "
            "BSE > Bulk/Block Deals"
        ),
        "how_to_check": (
            "NSE/BSE bulk deal data; "
            "Trendlyne > Bulk Deals"
        ),
    }

    # 13.5 Analyst Coverage
    num_analysts = self.safe_get("numberOfAnalystOpinions")
    target_price = self.safe_get("targetMeanPrice")
    rec = self.safe_get("recommendationKey", "N/A")
    analyst_score = 5
    if num_analysts is not None and num_analysts > 15:
        analyst_score = 7
    elif num_analysts is not None and num_analysts > 5:
        analyst_score = 6
    analyst_value = "N/A"
    if num_analysts is not None and target_price is not None:
        analyst_value = (
            "Analysts: {}, Consensus: {}, Target: Rs{:.0f}"
        ).format(num_analysts, rec, target_price)
    results["13.5 Analyst Coverage"] = {
        "sub_weight": 0.4,
        "value": analyst_value,
        "score": analyst_score,
        "source": (
            "Yahoo Finance - numberOfAnalystOpinions, "
            "recommendationKey"
        ),
        "how_to_check": (
            "Trendlyne > Forecaster; "
            "MoneyControl > Analyst Recommendations"
        ),
    }

    results["13.6 Index Inclusion"] = {
        "sub_weight": 0.4,
        "value": "Check NSE index constituents",
        "score": 5,
        "source": (
            "https://www.niftyindices.com/ > Index constituents"
        ),
        "how_to_check": (
            "Check if stock is in "
            "Nifty50/Next50/Midcap150/MSCI India"
        ),
    }

    results["13.7 Short Interest / Futures OI"] = {
        "sub_weight": 0.5,
        "value": "Check F&O data on NSE",
        "score": 5,
        "source": (
            "NSE > F&O > OI data; Trendlyne > Futures OI"
        ),
        "how_to_check": (
            "NSE > Derivatives > OI; "
            "Check if long buildup or short buildup"
        ),
    }

    float_pct = self.safe_get("floatShares")
    float_value = "N/A"
    if float_pct is not None:
        float_value = "Float Shares: {:,.0f}".format(float_pct)
    results["13.8 Free Float"] = {
        "sub_weight": 0.5,
        "value": float_value,
        "score": 5,
        "source": "Yahoo Finance - floatShares",
        "how_to_check": (
            "BSE > Shareholding Pattern > Public holding %; "
            "Higher free float = better liquidity"
        ),
    }

    return results

def analyze_technicals(self):
    """Section 14: Technical & Price Factors (4%)."""
    results = {}
    hist = self.f.history

    if hist.empty or len(hist) < 50:
        default_names = [
            "14.1 Trend Analysis (DMA)",
            "14.2 Volume Profile",
            "14.3 RSI / Momentum Indicators",
            "14.4 Support / Resistance",
            "14.5 52W High/Low Proximity",
            "14.6 Relative Strength vs Nifty",
            "14.7 Chart Patterns",
            "14.8 Volatility (Beta)",
            "14.9 Delivery % Trends",
        ]
        for name in default_names:
            results[name] = {
                "sub_weight": 0.5,
                "value": "Insufficient data",
                "score": 5,
                "source": "N/A",
                "how_to_check": (
                    "Need at least 50 days of price history"
                ),
            }
        return results

    close = hist["Close"]
    volume = hist["Volume"]
    current_price = close.iloc[-1]

    # 14.1 Trend Analysis - DMA
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = None
    if len(close) >= 200:
        sma200 = close.rolling(200).mean().iloc[-1]

    trend_score = 5
    trend_text = "CMP: Rs{:.1f}, 50DMA: Rs{:.1f}".format(
        current_price, sma50
    )
    if sma200 is not None:
        trend_text += ", 200DMA: Rs{:.1f}".format(sma200)
        if current_price > sma50 > sma200:
            trend_score = 8
            trend_text += " | BULLISH (Price > 50DMA > 200DMA)"
        elif current_price > sma200:
            trend_score = 6
            trend_text += " | MODERATE BULLISH"
        elif current_price < sma50 < sma200:
            trend_score = 2
            trend_text += " | BEARISH (Price < 50DMA < 200DMA)"
        else:
            trend_score = 4
            trend_text += " | MIXED"
    results["14.1 Trend Analysis (DMA)"] = {
        "sub_weight": 0.7,
        "value": trend_text,
        "score": trend_score,
        "source": (
            "Calculated from Yahoo Finance daily close "
            "prices (50/200 DMA)"
        ),
        "how_to_check": (
            "TradingView.com > Add 50/200 SMA; "
            "Chartink.com"
        ),
    }

    # 14.2 Volume Profile
    avg_vol_20 = volume.tail(20).mean()
    avg_vol_50 = volume.tail(50).mean()
    vol_score = 5
    vol_text = "20D Avg Vol: {:,.0f}, 50D Avg: {:,.0f}".format(
        avg_vol_20, avg_vol_50
    )
    if avg_vol_20 > avg_vol_50 * 1.3:
        vol_score = 7
        vol_text += " | Volume EXPANDING"
    elif avg_vol_20 < avg_vol_50 * 0.7:
        vol_score = 4
        vol_text += " | Volume CONTRACTING"
    results["14.2 Volume Profile"] = {
        "sub_weight": 0.5,
        "value": vol_text,
        "score": vol_score,
        "source": (
            "Yahoo Finance daily volume data - "
            "20D and 50D average"
        ),
        "how_to_check": (
            "TradingView; NSE historical data; "
            "ChartInk volume scans"
        ),
    }

    # 14.3 RSI / Momentum
    rsi_indicator = ta_lib.momentum.RSIIndicator(
        close, window=14
    )
    rsi_series = rsi_indicator.rsi()
    rsi = None
    if not rsi_series.empty:
        rsi = rsi_series.iloc[-1]

    macd_indicator = ta_lib.trend.MACD(close)
    macd_line = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]

    rsi_score = 5
    rsi_text = "N/A"
    if rsi is not None:
        rsi_text = "RSI(14): {:.1f}, MACD: {:.2f}, Signal: {:.2f}".format(
            rsi, macd_line, macd_signal
        )
        if 40 < rsi < 60:
            rsi_score = 5
        elif rsi < 30:
            rsi_score = 8
            rsi_text += " | OVERSOLD"
        elif rsi < 40:
            rsi_score = 6
        elif rsi > 70:
            rsi_score = 3
            rsi_text += " | OVERBOUGHT"
        elif rsi > 60:
            rsi_score = 5
        if macd_line > macd_signal:
            rsi_score = min(10, rsi_score + 1)
            rsi_text += " | MACD Bullish"
    results["14.3 RSI / Momentum Indicators"] = {
        "sub_weight": 0.4,
        "value": rsi_text,
        "score": rsi_score,
        "source": (
            "Calculated using ta library from Yahoo Finance "
            "price data (RSI-14, MACD 12-26-9)"
        ),
        "how_to_check": (
            "TradingView > Indicators > RSI, MACD; "
            "Chartink > Technical scans"
        ),
    }

    # 14.4 Support / Resistance
    if len(close) >= 252:
        high_52w = close.tail(252).max()
        low_52w = close.tail(252).min()
    else:
        high_52w = close.max()
        low_52w = close.min()
    results["14.4 Support / Resistance"] = {
        "sub_weight": 0.4,
        "value": (
            "52W High: Rs{:.1f}, 52W Low: Rs{:.1f}, "
            "CMP: Rs{:.1f}"
        ).format(high_52w, low_52w, current_price),
        "score": 5,
        "source": "Yahoo Finance 52-week price data",
        "how_to_check": (
            "TradingView > Draw support/resistance; "
            "NSE > 52W H/L"
        ),
    }

    # 14.5 52W Proximity
    range_52w = high_52w - low_52w
    prox_score = 5
    prox_text = "N/A"
    if range_52w > 0:
        position = (current_price - low_52w) / range_52w
        if position > 0.9:
            prox_score = 4
            prox_text = "Near 52W High ({:.0f}% of range)".format(
                position * 100
            )
        elif position > 0.6:
            prox_score = 6
            prox_text = "Upper half of range ({:.0f}%)".format(
                position * 100
            )
        elif position > 0.3:
            prox_score = 7
            prox_text = "Mid-range ({:.0f}%) - potential value".format(
                position * 100
            )
        else:
            prox_score = 5
            prox_text = (
                "Near 52W Low ({:.0f}%) - check fundamentals"
            ).format(position * 100)
    results["14.5 52W High/Low Proximity"] = {
        "sub_weight": 0.3,
        "value": prox_text,
        "score": prox_score,
        "source": (
            "Calculated from Yahoo Finance 52-week price range"
        ),
        "how_to_check": (
            "NSE/BSE > Company > 52 Week H/L"
        ),
    }

    # 14.6 Relative Strength vs Nifty
    beta = self.safe_get("beta")
    rs_score = 5
    if beta is not None:
        if 0.8 <= beta <= 1.2:
            rs_score = 6
        elif beta < 0.8:
            rs_score = 5
        else:
            rs_score = 5
    rs_value = "N/A"
    if beta is not None:
        rs_value = "Beta: {:.2f}".format(beta)
    results["14.6 Relative Strength vs Nifty"] = {
        "sub_weight": 0.4,
        "value": rs_value,
        "score": rs_score,
        "source": "Yahoo Finance - beta (vs market)",
        "how_to_check": (
            "TradingView > Compare with NIFTY; "
            "Tickertape > Beta; "
            "StockEdge > Relative Strength"
        ),
    }

    # 14.7 Chart Patterns
    results["14.7 Chart Patterns"] = {
        "sub_weight": 0.3,
        "value": "Visual analysis needed",
        "score": 5,
        "source": "TradingView.com chart analysis",
        "how_to_check": (
            "TradingView.com > Draw tools; "
            "Look for Cup&Handle, H&S, triangles, flags"
        ),
    }

    # 14.8 Volatility (Beta)
    returns = close.pct_change().dropna()
    vol = returns.std() * np.sqrt(252) * 100
    vol_score2 = 5
    if vol < 25:
        vol_score2 = 7
    elif vol < 40:
        vol_score2 = 5
    else:
        vol_score2 = 3
    vol_value = "Annualized Vol: {:.1f}%".format(vol)
    if beta is not None:
        vol_value = (
            "Annualized Volatility: {:.1f}%, Beta: {:.2f}"
        ).format(vol, beta)
    results["14.8 Volatility (Beta)"] = {
        "sub_weight": 0.5,
        "value": vol_value,
        "score": vol_score2,
        "source": (
            "Calculated from Yahoo Finance daily returns "
            "(annualized std dev)"
        ),
        "how_to_check": (
            "Tickertape > Risk metrics; "
            "StockEdge > Volatility"
        ),
    }

    # 14.9 Delivery % Trends
    results["14.9 Delivery % Trends"] = {
        "sub_weight": 0.5,
        "value": "Check NSE delivery data",
        "score": 5,
        "source": (
            "NSE > Historical Data > "
            "Security-wise Delivery Position"
        ),
        "how_to_check": (
            "NSE website > Market Data > Delivery data; "
            "Chartink > Delivery % scans; "
            ">50% consistently = conviction"
        ),
    }

    return results

# =============================================================================
# SECTOR-SPECIFIC ADJUSTMENTS
# =============================================================================

def get_sector_adjustments(sector):
"""Return sector-specific additional factors from Part C."""
adjustments = {
    "banking": {
        "NPA / Asset Quality (GNPA, NNPA, PCR)": {
            "weight": 3.0,
            "score": 5,
            "source": (
                "Quarterly results > Asset Quality slide; "
                "RBI DBIE database"
            ),
            "how_to_check": (
                "Screener.in > Company > Asset Quality; "
                "Trendlyne > Banking Ratios; GNPA<3%=Good"
            ),
        },
        "NIM / Spread": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "Quarterly investor presentation; Screener.in"
            ),
            "how_to_check": (
                "NIM > 3% for banks = Good; "
                "Check trend over 4 quarters"
            ),
        },
        "CASA Ratio": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Bank quarterly results; "
                "Investor presentation"
            ),
            "how_to_check": (
                "CASA > 40% = Good; > 50% = Excellent"
            ),
        },
        "CAR (Capital Adequacy)": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Basel III disclosure; Quarterly results"
            ),
            "how_to_check": (
                "CET-1 > 10%, Total CAR > 14% = Comfortable"
            ),
        },
        "ALM Mismatch": {
            "weight": 0.5,
            "score": 5,
            "source": "Annual Report > ALM disclosure",
            "how_to_check": (
                "Check if short-term liabilities > "
                "short-term assets"
            ),
        },
    },
    "it": {
        "Attrition Rate": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Quarterly results; HR section of "
                "investor presentation"
            ),
            "how_to_check": (
                "LTM Attrition < 15% = Good; "
                "> 20% = Concern"
            ),
        },
        "Deal Wins / TCV": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "Quarterly earnings release; "
                "Investor presentation > Deal wins slide"
            ),
            "how_to_check": (
                "Large deal TCV growing QoQ; "
                "Book-to-bill > 1.2x = healthy"
            ),
        },
        "Utilization Rate": {
            "weight": 1.0,
            "score": 5,
            "source": "Quarterly investor presentation",
            "how_to_check": (
                "Utilization > 82% = Good; "
                "Including trainees > 78%"
            ),
        },
        "Subcontracting Cost": {
            "weight": 0.5,
            "score": 5,
            "source": (
                "P&L > Subcontracting expense line"
            ),
            "how_to_check": (
                "If increasing as % of revenue "
                "= margin pressure"
            ),
        },
    },
    "pharma": {
        "USFDA Regulatory Actions": {
            "weight": 3.0,
            "score": 5,
            "source": (
                "https://www.fda.gov/ > Inspections database; "
                "Company announcements"
            ),
            "how_to_check": (
                "Check FDA Warning Letters, Form 483s, "
                "Import Alerts for company plants"
            ),
        },
        "ANDA Pipeline / Product Launches": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "FDA Orange Book; "
                "Company investor presentation"
            ),
            "how_to_check": (
                "Number of ANDAs filed/approved; "
                "First-to-file opportunities"
            ),
        },
        "DPCO / Price Control": {
            "weight": 1.5,
            "score": 5,
            "source": "NPPA website; NLEM list",
            "how_to_check": (
                "What % of revenue is under price control? "
                "High % = margin risk"
            ),
        },
        "API Dependency on China": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Annual Report > RM sourcing; "
                "Industry reports"
            ),
            "how_to_check": (
                "Higher domestic API sourcing = better; "
                "Check PLI for bulk drugs"
            ),
        },
    },
    "energy": {
        "Natural Gas Prices (APM/spot)": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "PPAC India; Kirit Parikh formula; "
                "Henry Hub/JKM prices"
            ),
            "how_to_check": (
                "APM price set by govt semi-annually; "
                "Check PPAC.gov.in"
            ),
        },
        "GRM (Gross Refining Margin)": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "Singapore GRM benchmark; "
                "Company quarterly results"
            ),
            "how_to_check": (
                "Company investor presentation > GRM; "
                "Singapore complex GRM as benchmark"
            ),
        },
        "Govt Subsidy / Under-recovery": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "PPAC; Company quarterly results; "
                "Budget allocation"
            ),
            "how_to_check": (
                "For OMCs - check if fuel prices are "
                "market-linked or controlled"
            ),
        },
        "Energy Transition Policy": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "MNRE; MoPNG; Ethanol blending targets"
            ),
            "how_to_check": (
                "Ethanol blending %, Green hydrogen policy, "
                "Company RE investments"
            ),
        },
    },
    "fmcg": {
        "Raw Material Costs (Palm Oil, Milk, etc.)": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "MCX commodity prices; "
                "Company quarterly results > RM cost"
            ),
            "how_to_check": (
                "Track palm oil (SEA India), milk prices; "
                "Check gross margin trend"
            ),
        },
        "Distribution Reach": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Investor presentation > Distribution slide"
            ),
            "how_to_check": (
                "Direct reach > 1M outlets = Strong; "
                "Numeric + Weighted distribution"
            ),
        },
        "Volume Growth vs Price Growth": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Quarterly earnings commentary; "
                "Analyst reports"
            ),
            "how_to_check": (
                "Volume growth > 5% = healthy; "
                "Only price-led growth = unsustainable"
            ),
        },
        "Brand Strength / Ad Spend": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Annual Report > A&P spend; "
                "Brand Finance rankings"
            ),
            "how_to_check": (
                "A&P as % of revenue; Brand recall surveys"
            ),
        },
        "Quick Commerce / D2C Disruption": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Industry reports; "
                "Company e-commerce % disclosure"
            ),
            "how_to_check": (
                "Is company adapting to quick commerce? "
                "D2C channel contribution?"
            ),
        },
    },
    "auto": {
        "Steel/Aluminum/Rubber Prices": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "LME prices; MCX; "
                "CRISIL commodity outlook"
            ),
            "how_to_check": (
                "Check RM cost as % of revenue "
                "in quarterly results"
            ),
        },
        "EV Transition Speed": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "SIAM/FADA data; "
                "Company EV strategy presentation"
            ),
            "how_to_check": (
                "Company EV portfolio; EV as % of total sales; "
                "Battery sourcing strategy"
            ),
        },
        "Monthly Sales Volume (SIAM/FADA)": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "https://www.siam.in/; "
                "https://www.fada.in/; "
                "Company monthly sales release"
            ),
            "how_to_check": (
                "Monthly dispatch and retail data; "
                "YoY growth; Inventory days at dealer"
            ),
        },
        "Emission Norms (BS-VII, CAFE)": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "MoRTH notifications; "
                "Company compliance disclosures"
            ),
            "how_to_check": (
                "Timeline for next emission norm; "
                "Company readiness and compliance cost"
            ),
        },
    },
    "metals": {
        "China Demand & Policy": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "China PMI; PBOC policy; "
                "China property sector data"
            ),
            "how_to_check": (
                "China construction PMI; "
                "Property starts; Steel production data"
            ),
        },
        "Mining Lease Renewals": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "MMDR Act; State mining department; "
                "Company disclosures"
            ),
            "how_to_check": (
                "Check lease expiry dates in Annual Report; "
                "Any auction risk?"
            ),
        },
        "Capacity Utilization": {
            "weight": 0.5,
            "score": 5,
            "source": (
                "Investor presentation; Industry data"
            ),
            "how_to_check": (
                "Higher utilization (>80%) = "
                "operating leverage benefit"
            ),
        },
    },
    "realestate": {
        "RERA Compliance": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "State RERA websites; "
                "Company project registrations"
            ),
            "how_to_check": (
                "Check RERA registered projects; "
                "Delivery track record"
            ),
        },
        "Land Bank Valuation": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Investor presentation > Land bank slide; "
                "Circle rates"
            ),
            "how_to_check": (
                "Total land bank in sq ft; "
                "Location quality; "
                "Book value vs market value"
            ),
        },
        "Pre-Sales / Booking Value": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Quarterly pre-sales data in "
                "investor presentation"
            ),
            "how_to_check": (
                "Pre-sales growing QoQ/YoY = strong demand; "
                "Booking value vs guidance"
            ),
        },
        "Cement/Steel/Labour Cost": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Construction cost index; "
                "Company quarterly results"
            ),
            "how_to_check": (
                "Construction cost per sq ft trend; "
                "Impact on margins"
            ),
        },
    },
    "telecom": {
        "ARPU Trend": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "Quarterly results; "
                "TRAI performance indicators"
            ),
            "how_to_check": (
                "ARPU > Rs200/month = Good for Airtel/Jio; "
                "Rising ARPU = Bullish"
            ),
        },
        "Subscriber Additions": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "TRAI subscriber data; "
                "Company quarterly results"
            ),
            "how_to_check": (
                "Net subscriber additions; "
                "Active subscriber ratio"
            ),
        },
        "Spectrum Costs & Auctions": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "DoT auction results; "
                "Company spectrum holding details"
            ),
            "how_to_check": (
                "Check annual spectrum payment obligations; "
                "Debt from spectrum auctions"
            ),
        },
        "5G Capex / ROI": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Investor presentation > 5G rollout; "
                "Capex guidance"
            ),
            "how_to_check": (
                "5G capex completed %; "
                "Monetization strategy; "
                "Enterprise 5G use cases"
            ),
        },
    },
    "power": {
        "CERC/SERC Tariff Orders": {
            "weight": 2.5,
            "score": 5,
            "source": (
                "CERC/State SERC tariff orders; "
                "Regulatory filings"
            ),
            "how_to_check": (
                "Check allowed ROE; Tariff petition status; "
                "Multi-year tariff orders"
            ),
        },
        "PLF/CUF": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "CEA monthly generation data; "
                "Company quarterly results"
            ),
            "how_to_check": (
                "Thermal PLF > 70% = Good; "
                "Solar CUF > 22% = Good"
            ),
        },
        "Coal Availability": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "CIL production data; "
                "Company fuel cost in results"
            ),
            "how_to_check": (
                "Coal stock days at plant; "
                "Linkage vs e-auction vs import mix"
            ),
        },
        "Discom Health": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "PFC Report on Utility Performance; "
                "UDAY dashboard"
            ),
            "how_to_check": (
                "AT&C losses of buyer discoms; "
                "Payment delay track record"
            ),
        },
    },
    "chemicals": {
        "China + 1 Tailwind": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "Industry reports; "
                "China environmental policy news"
            ),
            "how_to_check": (
                "Company benefiting from supply shift "
                "from China? New customer wins?"
            ),
        },
        "Product Registration / Approval": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Company investor presentation; "
                "REACH/EPA registration status"
            ),
            "how_to_check": (
                "Number of registered products; "
                "Long registration = moat"
            ),
        },
        "Customer Stickiness / Qualification": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Investor presentation; Industry analysis"
            ),
            "how_to_check": (
                "Long qualification period for products = "
                "switching cost moat"
            ),
        },
        "Capex-to-Revenue Conversion": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Capex announcements vs revenue ramp; "
                "Investor presentations"
            ),
            "how_to_check": (
                "Track capex announced 2-3 years ago; "
                "Has it translated to revenue?"
            ),
        },
    },
    "cement": {
        "Limestone Reserve Life": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Annual Report > Mining section; "
                "Investor presentation"
            ),
            "how_to_check": (
                "Reserve life > 30 years = comfortable"
            ),
        },
        "Power & Fuel Cost": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Quarterly results; "
                "Cost breakup in investor presentation"
            ),
            "how_to_check": (
                "Fuel cost per ton trend; "
                "Petcoke vs coal mix; WHRS/solar usage %"
            ),
        },
        "Freight Cost": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Quarterly results; Cost breakup"
            ),
            "how_to_check": (
                "Freight per ton; Rail vs road mix; "
                "Lead distance management"
            ),
        },
        "Per-Ton Realization & EBITDA": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Quarterly investor presentation > "
                "Per ton metrics"
            ),
            "how_to_check": (
                "EBITDA/ton > Rs1000 = Good; "
                "Realization trend regional"
            ),
        },
    },
    "defence": {
        "Defence Budget Allocation": {
            "weight": 3.0,
            "score": 6,
            "source": (
                "Union Budget > Defence capital outlay; "
                "MoD Annual Report"
            ),
            "how_to_check": (
                "Capital outlay growing >10% YoY = Bullish; "
                "Modernization fund allocation"
            ),
        },
        "Indigenization / Make in India": {
            "weight": 2.0,
            "score": 6,
            "source": (
                "MoD > Positive Indigenization Lists; "
                "Defence Acquisition Procedure"
            ),
            "how_to_check": (
                "Is company product on positive list? "
                "Import embargo items?"
            ),
        },
        "Export Potential": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Defence export data; SIPRI database; "
                "Company order book geography"
            ),
            "how_to_check": (
                "India defence exports growing; "
                "Company export orders / LOIs"
            ),
        },
    },
    "textiles": {
        "Cotton / Yarn Prices": {
            "weight": 2.0,
            "score": 5,
            "source": (
                "MCX Cotton; "
                "CAI (Cotton Association of India) data"
            ),
            "how_to_check": (
                "Cotton prices stable/declining = "
                "margin benefit for spinners/garment"
            ),
        },
        "FTAs & Duty Benefits": {
            "weight": 1.5,
            "score": 5,
            "source": (
                "Commerce Ministry; "
                "India-EU/UK FTA progress"
            ),
            "how_to_check": (
                "FTA with major markets = export boost; "
                "GSP benefits status"
            ),
        },
        "Bangladesh/Vietnam Competition": {
            "weight": 1.0,
            "score": 5,
            "source": (
                "Industry reports; WTO trade data; "
                "Labour cost comparisons"
            ),
            "how_to_check": (
                "India cost competitiveness vs Bangladesh; "
                "Any shifts in buyer preferences?"
            ),
        },
    },
}
return adjustments.get(sector, {})

# =============================================================================
# SCORE CALCULATOR
# =============================================================================

def calculate_composite_score(macro_scores, micro_sections, sector_adj):
"""Calculate final composite score based on the framework."""

# Calculate macro weighted score
macro_total_weight = 0
macro_weighted_sum = 0
macro_breakdown = {}

for category, data in macro_scores.items():
    cat_weight = data["weight"]
    cat_score_sum = 0
    cat_sub_weight_sum = 0
    for factor, fdata in data["factors"].items():
        sw = fdata["sub_weight"]
        sc = fdata.get("score", fdata["default_score"])
        cat_score_sum += sw * sc
        cat_sub_weight_sum += sw
    if cat_sub_weight_sum > 0:
        cat_avg = cat_score_sum / cat_sub_weight_sum
    else:
        cat_avg = 5
    macro_breakdown[category] = {
        "weight": cat_weight,
        "avg_score": cat_avg,
    }
    macro_weighted_sum += cat_weight * cat_avg
    macro_total_weight += cat_weight

if macro_total_weight > 0:
    macro_score_out_of_10 = macro_weighted_sum / macro_total_weight
else:
    macro_score_out_of_10 = 5

# Calculate micro weighted score
micro_total_weight = 0
micro_weighted_sum = 0
micro_weights = {
    "7_financials": 15,
    "8_valuation": 8,
    "9_promoter": 10,
    "10_legal": 5,
    "11_competitive": 7,
    "12_growth": 7,
    "13_shareholding": 4,
    "14_technicals": 4,
}
micro_breakdown = {}

for section_key, section_data in micro_sections.items():
    section_weight = micro_weights.get(section_key, 5)
    sec_score_sum = 0
    sec_sub_weight_sum = 0
    for factor, fdata in section_data.items():
        sw = fdata["sub_weight"]
        sc = fdata["score"]
        sec_score_sum += sw * sc
        sec_sub_weight_sum += sw
    if sec_sub_weight_sum > 0:
        sec_avg = sec_score_sum / sec_sub_weight_sum
    else:
        sec_avg = 5
    micro_breakdown[section_key] = {
        "weight": section_weight,
        "avg_score": sec_avg,
    }
    micro_weighted_sum += section_weight * sec_avg
    micro_total_weight += section_weight

if micro_total_weight > 0:
    micro_score_out_of_10 = micro_weighted_sum / micro_total_weight
else:
    micro_score_out_of_10 = 5

# Sector adjustment bonus
sector_bonus = 0
for factor, fdata in sector_adj.items():
    w = fdata["weight"]
    s = fdata["score"]
    sector_bonus += w * (s - 5) / 10

# Final composite: 40% macro + 60% micro (out of 10) -> to 100
base_score = (
    0.4 * macro_score_out_of_10 + 0.6 * micro_score_out_of_10
) * 10
final_score = max(0, min(100, base_score + sector_bonus))

return {
    "macro_score_10": macro_score_out_of_10,
    "micro_score_10": micro_score_out_of_10,
    "macro_breakdown": macro_breakdown,
    "micro_breakdown": micro_breakdown,
    "sector_bonus": sector_bonus,
    "base_score": base_score,
    "final_score": final_score,
}

def get_recommendation(score):
"""Get recommendation label and emoji based on score."""
for (low, high), (rec, emoji) in SCORE_INTERPRETATION.items():
    if low <= score <= high:
        return rec, emoji
return "N/A", "?"

# =============================================================================
# EXCEL EXPORT
# =============================================================================

def generate_excel(stock_results):
"""Generate comprehensive Excel report."""
output = BytesIO()

with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    workbook = writer.book

    # Formats
    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#1F4E79",
        "font_color": "white",
        "border": 1,
        "text_wrap": True,
        "valign": "vcenter",
    })
    score_green = workbook.add_format({
        "bg_color": "#C6EFCE",
        "border": 1,
    })
    score_yellow = workbook.add_format({
        "bg_color": "#FFEB9C",
        "border": 1,
    })
    score_red = workbook.add_format({
        "bg_color": "#FFC7CE",
        "border": 1,
    })

    for stock_name, result in stock_results.items():
        # Summary sheet
        safe_name = stock_name.replace("/", "_").replace("\\", "_")
        if len(safe_name) > 28:
            sheet_sum = safe_name[:28] + "_S"
        else:
            sheet_sum = safe_name + "_Sum"

        summary_data = {
            "Metric": [
                "Stock",
                "Sector",
                "Final Score",
                "Recommendation",
                "Macro Score (out of 10)",
                "Micro Score (out of 10)",
                "Sector Adjustment",
                "CMP",
                "Market Cap",
                "PE (TTM)",
                "PE (Forward)",
                "ROE",
                "Debt/Equity",
                "52W High",
                "52W Low",
                "Beta",
            ],
            "Value": [
                stock_name,
                result.get("sector", "N/A"),
                "{:.1f}/100".format(
                    result["composite"]["final_score"]
                ),
                result["recommendation"][0],
                "{:.2f}".format(
                    result["composite"]["macro_score_10"]
                ),
                "{:.2f}".format(
                    result["composite"]["micro_score_10"]
                ),
                "{:.2f}".format(
                    result["composite"]["sector_bonus"]
                ),
                str(result["info"].get("currentPrice", "N/A")),
                str(result["info"].get("marketCap", "N/A")),
                str(result["info"].get("trailingPE", "N/A")),
                str(result["info"].get("forwardPE", "N/A")),
                str(result["info"].get("returnOnEquity", "N/A")),
                str(result["info"].get("debtToEquity", "N/A")),
                str(result["info"].get(
                    "fiftyTwoWeekHigh", "N/A"
                )),
                str(result["info"].get(
                    "fiftyTwoWeekLow", "N/A"
                )),
                str(result["info"].get("beta", "N/A")),
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(
            writer, sheet_name=sheet_sum, index=False
        )
        ws = writer.sheets[sheet_sum]
        ws.set_column("A:A", 25)
        ws.set_column("B:B", 30)

        # Detailed Scores sheet
        if len(safe_name) > 26:
            detail_name = safe_name[:26] + "_D"
        else:
            detail_name = safe_name + "_Det"

        all_factors = []

        # Macro factors
        for category, cat_data in result["macro_data"].items():
            for factor, fdata in cat_data["factors"].items():
                all_factors.append({
                    "Section": "MACRO",
                    "Category": category,
                    "Factor": factor,
                    "Sub-Weight (%)": fdata["sub_weight"],
                    "Score (1-10)": fdata.get(
                        "score", fdata.get("default_score", 5)
                    ),
                    "Value/Data": fdata.get("description", ""),
                    "Data Source": fdata.get("source", ""),
                    "How to Check": fdata.get(
                        "how_to_check", ""
                    ),
                })

        # Micro factors
        for section_key, section_data in (
            result["micro_sections"].items()
        ):
            for factor, fdata in section_data.items():
                all_factors.append({
                    "Section": "MICRO",
                    "Category": section_key,
                    "Factor": factor,
                    "Sub-Weight (%)": fdata["sub_weight"],
                    "Score (1-10)": fdata["score"],
                    "Value/Data": fdata.get("value", ""),
                    "Data Source": fdata.get("source", ""),
                    "How to Check": fdata.get(
                        "how_to_check", ""
                    ),
                })

        # Sector-specific factors
        for factor, fdata in result.get(
            "sector_adj", {}
        ).items():
            all_factors.append({
                "Section": "SECTOR-SPECIFIC",
                "Category": result.get("sector", ""),
                "Factor": factor,
                "Sub-Weight (%)": fdata["weight"],
                "Score (1-10)": fdata["score"],
                "Value/Data": "",
                "Data Source": fdata.get("source", ""),
                "How to Check": fdata.get("how_to_check", ""),
            })

        df_detail = pd.DataFrame(all_factors)
        df_detail.to_excel(
            writer, sheet_name=detail_name, index=False
        )
        ws2 = writer.sheets[detail_name]
        ws2.set_column("A:A", 12)
        ws2.set_column("B:B", 35)
        ws2.set_column("C:C", 40)
        ws2.set_column("D:D", 12)
        ws2.set_column("E:E", 10)
        ws2.set_column("F:F", 50)
        ws2.set_column("G:G", 60)
        ws2.set_column("H:H", 60)

        # Apply header format
        for col_num, value in enumerate(df_detail.columns):
            ws2.write(0, col_num, value, header_fmt)

        # Conditional formatting on score column (col E = index 4)
        num_rows = len(df_detail)
        ws2.conditional_format(1, 4, num_rows, 4, {
            "type": "cell",
            "criteria": ">=",
            "value": 7,
            "format": score_green,
        })
        ws2.conditional_format(1, 4, num_rows, 4, {
            "type": "cell",
            "criteria": "between",
            "minimum": 4,
            "maximum": 6,
            "format": score_yellow,
        })
        ws2.conditional_format(1, 4, num_rows, 4, {
            "type": "cell",
            "criteria": "<",
            "value": 4,
            "format": score_red,
        })

output.seek(0)
return output

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
"""Main Streamlit application."""
st.set_page_config(
    page_title="Indian Stock Analyzer - 150+ Factor Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Comprehensive Indian Stock Analyzer")
st.markdown(
    "**150+ Factor Analysis Framework for Indian Equities**"
)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### Enter Stock Symbols")
    st.markdown(
        "*Use NSE symbols (e.g., RELIANCE, TCS, HDFCBANK)*"
    )
    stock_input = st.text_area(
        "Stock Symbols (one per line or comma-separated)",
        value="RELIANCE\nTCS",
        height=100,
    )

    sector = st.selectbox(
        "Select Sector", list(SECTOR_MAP.keys())
    )

    st.markdown("---")
    st.markdown("### Macro Score Override")
    st.markdown(
        "*Macro scores are market-wide. "
        "Adjust if you have current data.*"
    )
    macro_override = st.checkbox(
        "Use default macro scores (recommended)", value=True
    )

    analyze_button = st.button(
        "🔍 **ANALYZE**",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown(
        """
        1. Enter one or more NSE stock symbols
        2. Select the appropriate sector
        3. Click **ANALYZE**
        4. Review scores across all categories
        5. Download the Excel report
        6. Use the **Data Sources** to verify and refine

        **Note:** Qualitative factors default to 5/10.
        Update them in the Excel for a more accurate score.
        """
    )

# Parse stock names
stocks = []
if stock_input:
    for s in stock_input.replace(",", "\n").split("\n"):
        s = s.strip().upper()
        if s:
            stocks.append(s)

if analyze_button and stocks:
    sector_key = SECTOR_MAP[sector]
    all_results = {}

    progress = st.progress(0)
    status = st.empty()

    for idx, stock_name in enumerate(stocks):
        status.markdown(
            "### ⏳ Analyzing **{}**... ({}/{})".format(
                stock_name, idx + 1, len(stocks)
            )
        )

        # Fetch data
        fetcher = StockDataFetcher(stock_name)
        success = fetcher.fetch_all()

        if not success:
            st.error(
                "Could not fetch data for {}. "
                "Check if the symbol is correct (NSE symbol).".format(
                    stock_name
                )
            )
            continue

        # Get macro data
        macro_data = MacroAnalyzer.get_macro_defaults()

        # Micro analysis
        micro = MicroAnalyzer(fetcher, sector_key)
        micro_sections = {
            "7_financials": micro.analyze_financials(),
            "8_valuation": micro.analyze_valuation(),
            "9_promoter": micro.analyze_promoter_management(),
            "10_legal": micro.analyze_legal(),
            "11_competitive": micro.analyze_competitive(),
            "12_growth": micro.analyze_growth(),
            "13_shareholding": micro.analyze_shareholding(),
            "14_technicals": micro.analyze_technicals(),
        }

        # Sector adjustments
        sector_adj = get_sector_adjustments(sector_key)

        # Calculate composite score
        composite = calculate_composite_score(
            macro_data, micro_sections, sector_adj
        )
        recommendation = get_recommendation(
            composite["final_score"]
        )

        all_results[stock_name] = {
            "sector": sector,
            "info": fetcher.info,
            "macro_data": macro_data,
            "micro_sections": micro_sections,
            "sector_adj": sector_adj,
            "composite": composite,
            "recommendation": recommendation,
            "data_sources": fetcher.data_sources,
        }

        progress.progress((idx + 1) / len(stocks))

    status.markdown("### ✅ Analysis Complete!")

    # Display Results
    if all_results:
        st.markdown("---")

        # Summary cards
        st.header("📋 Summary")
        cols = st.columns(min(len(all_results), 4))
        for i, (stock, result) in enumerate(
            all_results.items()
        ):
            with cols[i % 4]:
                score = result["composite"]["final_score"]
                rec, emoji = result["recommendation"]
                if score >= 70:
                    color = "#28a745"
                elif score >= 55:
                    color = "#ffc107"
                else:
                    color = "#dc3545"
                card_html = (
                    '<div style="border: 2px solid {color}; '
                    'border-radius: 10px; padding: 15px; '
                    'text-align: center; margin: 5px;">'
                    '<h3>{stock}</h3>'
                    '<h1 style="color: {color};">{score:.1f}</h1>'
                    '<h4>{emoji} {rec}</h4>'
                    '<p>Macro: {macro:.1f}/10 | '
                    'Micro: {micro:.1f}/10</p>'
                    '</div>'
                ).format(
                    color=color,
                    stock=stock,
                    score=score,
                    emoji=emoji,
                    rec=rec,
                    macro=result["composite"]["macro_score_10"],
                    micro=result["composite"]["micro_score_10"],
                )
                st.markdown(card_html, unsafe_allow_html=True)

        # Detailed results per stock
        for stock_name, result in all_results.items():
            st.markdown("---")
            st.header(
                "📈 {} — Detailed Analysis".format(stock_name)
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏠 Overview",
                "🌍 Macro Scores",
                "🔬 Micro Scores",
                "🏭 Sector-Specific",
                "📚 Data Sources",
            ])

            with tab1:
                info = result["info"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "CMP",
                        "Rs{}".format(
                            info.get("currentPrice", "N/A")
                        ),
                    )
                    mcap = info.get("marketCap", 0)
                    if mcap and mcap > 0:
                        mcap_str = "Rs{:,.0f} Cr".format(
                            mcap / 1e7
                        )
                    else:
                        mcap_str = "N/A"
                    st.metric("Market Cap", mcap_str)
                with col2:
                    st.metric(
                        "PE (TTM)",
                        str(info.get("trailingPE", "N/A")),
                    )
                    st.metric(
                        "PE (Forward)",
                        str(info.get("forwardPE", "N/A")),
                    )
                with col3:
                    roe_val = info.get("returnOnEquity")
                    if roe_val is not None:
                        roe_str = "{:.1f}%".format(roe_val * 100)
                    else:
                        roe_str = "N/A"
                    st.metric("ROE", roe_str)
                    de_val = info.get("debtToEquity")
                    if de_val is not None:
                        de_str = "{:.2f}".format(de_val)
                    else:
                        de_str = "N/A"
                    st.metric("D/E Ratio", de_str)
                with col4:
                    st.metric(
                        "52W High",
                        "Rs{}".format(
                            info.get("fiftyTwoWeekHigh", "N/A")
                        ),
                    )
                    st.metric(
                        "52W Low",
                        "Rs{}".format(
                            info.get("fiftyTwoWeekLow", "N/A")
                        ),
                    )

                # Score breakdown chart
                comp = result["composite"]
                micro_bk = comp["micro_breakdown"]
                categories = list(micro_bk.keys())
                cat_labels = {
                    "7_financials": "Financials",
                    "8_valuation": "Valuation",
                    "9_promoter": "Promoter/Mgmt",
                    "10_legal": "Legal",
                    "11_competitive": "Competition",
                    "12_growth": "Growth",
                    "13_shareholding": "Shareholding",
                    "14_technicals": "Technicals",
                }

                bar_colors = []
                bar_texts = []
                bar_x = []
                bar_y = []
                for c in categories:
                    avg = micro_bk[c]["avg_score"]
                    bar_x.append(cat_labels.get(c, c))
                    bar_y.append(avg)
                    bar_texts.append("{:.1f}".format(avg))
                    if avg >= 6:
                        bar_colors.append("#2E86AB")
                    elif avg >= 4:
                        bar_colors.append("#F6AE2D")
                    else:
                        bar_colors.append("#E94F37")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bar_x,
                    y=bar_y,
                    marker_color=bar_colors,
                    text=bar_texts,
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Micro Score Breakdown - {}".format(
                        stock_name
                    ),
                    yaxis_title="Score (out of 10)",
                    yaxis_range=[0, 10],
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Macro Score Breakdown")
                st.markdown(
                    "*These scores are market-wide defaults. "
                    "Adjust in Excel for precision.*"
                )
                for category, cat_data in (
                    result["macro_data"].items()
                ):
                    with st.expander(
                        "📌 " + category, expanded=False
                    ):
                        rows = []
                        for factor, fdata in (
                            cat_data["factors"].items()
                        ):
                            rows.append({
                                "Factor": factor,
                                "Weight (%)": fdata["sub_weight"],
                                "Score (1-10)": fdata.get(
                                    "score",
                                    fdata["default_score"],
                                ),
                                "Description": (
                                    fdata["description"]
                                ),
                                "Source": fdata["source"],
                                "How to Check": (
                                    fdata["how_to_check"]
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            with tab3:
                st.subheader("Micro Score Breakdown")
                section_names = {
                    "7_financials": (
                        "7. Financial Performance (15%)"
                    ),
                    "8_valuation": "8. Valuation (8%)",
                    "9_promoter": (
                        "9. Promoter & Management (10%)"
                    ),
                    "10_legal": (
                        "10. Legal & Litigation (5%)"
                    ),
                    "11_competitive": (
                        "11. Competitive Positioning (7%)"
                    ),
                    "12_growth": "12. Growth Drivers (7%)",
                    "13_shareholding": (
                        "13. Shareholding (4%)"
                    ),
                    "14_technicals": "14. Technicals (4%)",
                }
                for section_key, section_data in (
                    result["micro_sections"].items()
                ):
                    with st.expander(
                        "📌 " + section_names.get(
                            section_key, section_key
                        ),
                        expanded=False,
                    ):
                        rows = []
                        for factor, fdata in (
                            section_data.items()
                        ):
                            rows.append({
                                "Factor": factor,
                                "Weight (%)": (
                                    fdata["sub_weight"]
                                ),
                                "Score (1-10)": fdata["score"],
                                "Value/Data": fdata.get(
                                    "value", ""
                                ),
                                "Source": fdata.get(
                                    "source", ""
                                ),
                                "How to Check": fdata.get(
                                    "how_to_check", ""
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            with tab4:
                st.subheader(
                    "Sector-Specific Factors: " + sector
                )
                if result["sector_adj"]:
                    rows = []
                    for factor, fdata in (
                        result["sector_adj"].items()
                    ):
                        rows.append({
                            "Factor": factor,
                            "Additional Weight (%)": (
                                fdata["weight"]
                            ),
                            "Score (1-10)": fdata["score"],
                            "Source": fdata.get("source", ""),
                            "How to Check": fdata.get(
                                "how_to_check", ""
                            ),
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.info(
                        "These scores default to 5 (Neutral). "
                        "Update in the downloaded Excel for "
                        "accurate assessment."
                    )
                else:
                    st.info(
                        "No sector-specific adjustments "
                        "for 'Other' sector."
                    )

            with tab5:
                st.subheader("Data Sources Reference")
                st.markdown(
                    "### API & Automated Data Sources"
                )
                for source_name, source_url in (
                    result["data_sources"].items()
                ):
                    st.markdown(
                        "- **{}**: {}".format(
                            source_name, source_url
                        )
                    )

                st.markdown(
                    "### Manual Verification Sources "
                    "(India-Specific)"
                )
                st.markdown(
                    """
| Source | URL | What It Provides |
|--------|-----|-----------------|
| **Screener.in** | https://www.screener.in/ | Financials, ratios, peers, documents |
| **Trendlyne** | https://trendlyne.com/ | Shareholding, forecaster, technicals |
| **Tickertape** | https://www.tickertape.in/ | Valuation, screeners, risk metrics |
| **Tijori Finance** | https://tijorifinance.com/ | Deep financials, industry comparison |
| **MoneyControl** | https://www.moneycontrol.com/ | News, financials, analyst recommendations |
| **BSE India** | https://www.bseindia.com/ | Shareholding pattern, corp announcements |
| **NSE India** | https://www.nseindia.com/ | F&O data, delivery data, corporate actions |
| **NSDL FPI Monitor** | https://www.fpi.nsdl.co.in/ | FII/FPI flow data |
| **RBI** | https://www.rbi.org.in/ | Monetary policy, credit data, inflation |
| **SEBI** | https://www.sebi.gov.in/ | Regulatory orders, mutual fund data |
| **TradingView** | https://www.tradingview.com/ | Charts, technical analysis |
| **Chartink** | https://chartink.com/ | Technical screeners |
| **Tofler** | https://www.tofler.in/ | MCA filings, director info |
| **Zauba Corp** | https://www.zaubacorp.com/ | Company registry, group structure |
| **IBEF** | https://www.ibef.org/ | Industry reports by sector |
                    """
                )

        # Excel Download
        st.markdown("---")
        st.header("📥 Download Excel Report")
        excel_data = generate_excel(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = "Stock_Analysis_{}_{}.xlsx".format(
            "_".join(stocks), timestamp
        )
        st.download_button(
            label="📥 Download Comprehensive Excel Report",
            data=excel_data,
            file_name=filename,
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )
        st.info(
            "Tip: Open the Excel and update qualitative scores "
            "(Legal, Competitive, Growth, Promoter sections) "
            "for a more accurate final score. All data sources "
            "and How to Check instructions are included."
        )

elif analyze_button and not stocks:
    st.warning("Please enter at least one stock symbol.")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; '
    'font-size: 0.9em;">'
    "<p><strong>Disclaimer:</strong> This tool is for "
    "educational and research purposes only. "
    "Not financial advice. Always do your own research "
    "and consult a SEBI-registered advisor before "
    "investing.</p>"
    "<p>Framework: 150+ Factor Comprehensive Stock Analysis "
    "| Built for Indian Markets</p>"
    "</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
main()
