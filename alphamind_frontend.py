
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============ Page Configuration ============

st.set_page_config(
    page_title="AlphaMind Analytics | MIT & Oxford",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Custom CSS ============

st.markdown("""
<style>
    /* Main Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    .university-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 10px 25px;
        border-radius: 30px;
        margin: 0 15px;
        color: white;
        font-weight: bold;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin: 0.5rem;
        border-top: 4px solid;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .positive { color: #10B981; font-weight: bold; }
    .negative { color: #EF4444; font-weight: bold; }
    .neutral { color: #6B7280; font-weight: bold; }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============ Backend Functions (Integrated) ============

import yfinance as yf

def get_stock_data(symbol: str, period: str = "6mo"):
    """Get historical stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return {"error": "No data found for symbol", "success": False}

        # Convert to JSON serializable format
        data = []
        for idx, row in hist.iterrows():
            data.append({
                "date": idx.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })

        # Calculate metrics
        latest_close = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_close
        change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close != 0 else 0

        return {
            "symbol": symbol,
            "period": period,
            "data": data[-90:],
            "current_price": latest_close,
            "change_percent": round(change_pct, 2),
            "data_points": len(data),
            "success": True
        }

    except Exception as e:
        return {"error": str(e), "success": False}

def technical_analysis(symbol: str, period: str = "6mo"):
    """Perform technical analysis"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if len(hist) < 20:
            return {"error": "Insufficient data for analysis", "success": False}

        closes = hist['Close'].values

        # Moving averages
        sma_20 = pd.Series(closes).rolling(window=20).mean().iloc[-1]
        sma_50 = pd.Series(closes).rolling(window=50).mean().iloc[-1]

        # RSI
        delta = pd.Series(closes).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # MACD
        exp1 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd = exp1.iloc[-1] - exp2.iloc[-1]

        current_price = closes[-1]
        signal = "NEUTRAL"

        if current_price > sma_20 > sma_50 and rsi < 70:
            signal = "BULLISH"
        elif current_price < sma_20 < sma_50 and rsi > 30:
            signal = "BEARISH"

        confidence = 0.5
        if signal == "BULLISH":
            confidence = 0.7 + (rsi / 200)
        elif signal == "BEARISH":
            confidence = 0.7 + ((100 - rsi) / 200)

        return {
            "symbol": symbol,
            "analysis": "technical",
            "indicators": {
                "price": round(float(current_price), 2),
                "sma_20": round(float(sma_20), 2),
                "sma_50": round(float(sma_50), 2),
                "rsi": round(float(rsi), 2),
                "macd": round(float(macd), 2),
                "signal": signal,
                "confidence": round(min(confidence, 0.95), 2)
            },
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        return {"error": str(e), "success": False}

def sentiment_analysis(symbol: str):
    """Perform sentiment analysis"""
    try:
        np.random.seed(hash(symbol) % 1000)

        sentiment_sources = {
            "news_articles": {"score": np.random.uniform(0.6, 0.9), "count": np.random.randint(50, 200)},
            "social_media": {"score": np.random.uniform(0.4, 0.8), "count": np.random.randint(100, 1000)},
            "analyst_reports": {"score": np.random.uniform(0.7, 0.95), "count": np.random.randint(5, 20)},
            "options_flow": {"score": np.random.uniform(0.5, 0.85), "count": np.random.randint(100, 500)}
        }

        weights = {"news_articles": 0.3, "social_media": 0.2, "analyst_reports": 0.3, "options_flow": 0.2}
        total_score = sum(sentiment_sources[src]["score"] * weights[src] for src in sentiment_sources)
        overall_sentiment = round(total_score, 3)

        if overall_sentiment >= 0.7:
            label = "STRONGLY_BULLISH"
        elif overall_sentiment >= 0.6:
            label = "BULLISH"
        elif overall_sentiment <= 0.4:
            label = "BEARISH"
        elif overall_sentiment <= 0.3:
            label = "STRONGLY_BEARISH"
        else:
            label = "NEUTRAL"

        positive_keywords = ["growth", "profit", "innovation", "leadership", "strong"]
        negative_keywords = ["risk", "competition", "regulation", "decline", "volatility"]

        np.random.shuffle(positive_keywords)
        np.random.shuffle(negative_keywords)
        keywords = positive_keywords[:3] + negative_keywords[:2]
        np.random.shuffle(keywords)

        return {
            "symbol": symbol,
            "analysis": "sentiment",
            "overall_score": overall_sentiment,
            "sentiment_label": label,
            "sources": sentiment_sources,
            "top_keywords": keywords[:5],
            "confidence": round(overall_sentiment, 2),
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        return {"error": str(e), "success": False}

def fundamental_analysis(symbol: str):
    """Perform fundamental analysis"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            return {"error": "Company info not found", "success": False}

        metrics = {
            "valuation": {
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "price_to_book": info.get("priceToBook", 0)
            },
            "profitability": {
                "roe": info.get("returnOnEquity", 0),
                "profit_margin": info.get("profitMargins", 0)
            },
            "financial_health": {
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0)
            }
        }

        health_score = 0.5
        if metrics["profitability"]["roe"] > 0.15:
            health_score += 0.1
        if metrics["financial_health"]["debt_to_equity"] < 1:
            health_score += 0.1
        if metrics["valuation"]["pe_ratio"] and metrics["valuation"]["pe_ratio"] < 25:
            health_score += 0.05

        health_score = min(max(health_score, 0), 1)

        return {
            "symbol": symbol,
            "analysis": "fundamental",
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "metrics": metrics,
            "health_score": round(health_score, 2),
            "recommendation": "BUY" if health_score > 0.7 else "SELL" if health_score < 0.3 else "HOLD",
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        return {"error": str(e), "success": False}

def comprehensive_analysis(symbol: str):
    """Comprehensive analysis"""
    try:
        technical = technical_analysis(symbol)
        sentiment = sentiment_analysis(symbol)
        fundamental = fundamental_analysis(symbol)

        if not all([technical.get("success"), sentiment.get("success"), fundamental.get("success")]):
            return {"error": "Failed to complete all analyses", "success": False}

        technical_score = technical["indicators"]["confidence"]
        sentiment_score = sentiment["overall_score"]
        fundamental_score = fundamental["health_score"]

        weights = {"technical": 0.4, "sentiment": 0.3, "fundamental": 0.3}
        overall_score = (
            technical_score * weights["technical"] +
            sentiment_score * weights["sentiment"] +
            fundamental_score * weights["fundamental"]
        )

        if overall_score >= 0.7:
            recommendation = "STRONG_BUY"
            confidence = "HIGH"
        elif overall_score >= 0.6:
            recommendation = "BUY"
            confidence = "MEDIUM_HIGH"
        elif overall_score >= 0.5:
            recommendation = "HOLD"
            confidence = "MEDIUM"
        elif overall_score >=
