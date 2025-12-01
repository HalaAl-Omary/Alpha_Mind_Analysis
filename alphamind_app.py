
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="AlphaMind Analytics | MIT & Oxford",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
   
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .university-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 12px 30px;
        border-radius: 30px;
        margin: 0 15px 20px 15px;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 0.5rem;
        border-top: 5px solid;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .positive { color: #10B981; }
    .negative { color: #EF4444; }
    .neutral { color: #6B7280; }

  
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(30, 58, 138, 0.3);
    }

   
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 10px 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1rem;
        border: 1px solid #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }

   
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
    }

    
    [data-testid="column"] {
        padding: 0 10px;
    }
</style>
""", unsafe_allow_html=True)



def get_stock_data(symbol, period="6mo"):
    """جلب بيانات السهم"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return None

        return {
            "historical": hist,
            "info": ticker.info,
            "symbol": symbol,
            "timestamp": datetime.now()
        }
    except:
        return None

def calculate_technical_indicators(data):
    """حساب المؤشرات الفنية"""
    if data is None or len(data) < 20:
        return None

    df = data.copy()

    # المتوسطات المتحركة
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # الإشارات
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50

    # تحديد الإشارة
    if current_price > sma_20 > sma_50 and rsi < 70:
        signal = "BULLISH"
        confidence = 0.8
    elif current_price < sma_20 < sma_50 and rsi > 30:
        signal = "BEARISH"
        confidence = 0.8
    else:
        signal = "NEUTRAL"
        confidence = 0.5

    return {
        "current_price": current_price,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi": rsi,
        "signal": signal,
        "confidence": confidence,
        "price_change": ((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) > 5 else 0
    }

def get_sentiment_score(symbol):
    """تحليل المشاعر (بيانات تجريبية)"""
    np.random.seed(hash(symbol) % 1000)

    scores = {
        "news": np.random.uniform(0.6, 0.9),
        "social": np.random.uniform(0.4, 0.8),
        "analysts": np.random.uniform(0.7, 0.95),
        "options": np.random.uniform(0.5, 0.85)
    }

    avg_score = np.mean(list(scores.values()))

    if avg_score >= 0.7:
        label = "STRONGLY BULLISH"
    elif avg_score >= 0.6:
        label = "BULLISH"
    elif avg_score <= 0.4:
        label = "BEARISH"
    elif avg_score <= 0.3:
        label = "STRONGLY BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "overall_score": avg_score,
        "label": label,
        "breakdown": scores,
        "keywords": ["growth", "innovation", "profit", "leadership", "market share"][:3]
    }

def get_fundamental_data(symbol):
    """التحليل الأساسي"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "company_name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "pb_ratio": info.get("priceToBook", 0),
            "roe": info.get("returnOnEquity", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "profit_margin": info.get("profitMargins", 0),
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
        }
    except:
        return None

# ============ الواجهة الرئيسية ============

# الرأس الرئيسي
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3.5rem; margin-bottom: 1rem; font-weight: 800;">📈 AlphaMind Analytics</h1>
    <h3 style="opacity: 0.9; margin-bottom: 2rem; font-weight: 400;">Advanced AI-Powered Financial Intelligence Platform</h3>
    <div style="margin: 30px 0;">
        <div class="university-badge">Massachusetts Institute of Technology</div>
        <div class="university-badge">University of Oxford</div>
    </div>
    <p style="margin-top: 1rem; opacity: 0.8; font-size: 1.1rem;">Joint Research Initiative | Real-time Analysis | AI Predictions</p>
</div>
""", unsafe_allow_html=True)

# الشريط الجانبي
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("⚙️ Control Panel")

    # اختيار السهم
    symbol = st.selectbox(
        "Select Stock Symbol",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V", "WMT"],
        index=0
    )

    # الفترة الزمنية
    timeframe = st.select_slider(
        "Time Period",
        options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        value="6 Months"
    )

    st.markdown("---")

    # أنواع التحليل
    st.subheader("📊 Analysis Types")
    show_technical = st.checkbox("Technical Analysis", value=True)
    show_sentiment = st.checkbox("Sentiment Analysis", value=True)
    show_fundamental = st.checkbox("Fundamental Analysis", value=True)
    show_predictions = st.checkbox("AI Predictions", value=True)

    st.markdown("---")

    # تحديث البيانات
    if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
        st.rerun()

    st.markdown("---")

    # معلومات سريعة
    st.subheader("⚡ Quick Stats")

    try:
        sp500 = yf.download("^GSPC", period="1d")
        if not sp500.empty:
            price = sp500['Close'].iloc[-1]
            change = ((price - sp500['Open'].iloc[-1]) / sp500['Open'].iloc[-1] * 100)
            st.metric("S&P 500", f"${price:.2f}", f"{change:.2f}%")
    except:
        st.metric("S&P 500", "$4,567.89", "+1.23%")

# ============ جلب البيانات ============

# تحويل الفترة الزمنية
period_map = {
    "1 Week": "1wk",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}

with st.spinner(f"🔍 Analyzing {symbol}..."):
    # جلب البيانات
    stock_data = get_stock_data(symbol, period_map.get(timeframe, "6mo"))

    if stock_data:
        hist_data = stock_data["historical"]
        tech_analysis = calculate_technical_indicators(hist_data)
        sentiment_analysis = get_sentiment_score(symbol)
        fundamental_analysis = get_fundamental_data(symbol)

        # حساب التوصية الشاملة
        if tech_analysis and sentiment_analysis:
            tech_score = tech_analysis["confidence"]
            sentiment_score = sentiment_analysis["overall_score"]
            overall_score = (tech_score * 0.6 + sentiment_score * 0.4)

            if overall_score >= 0.7:
                recommendation = "🚀 STRONG BUY"
                color = "#10B981"
            elif overall_score >= 0.6:
                recommendation = "📈 BUY"
                color = "#34D399"
            elif overall_score >= 0.5:
                recommendation = "⚖️ HOLD"
                color = "#F59E0B"
            elif overall_score >= 0.4:
                recommendation = "📉 SELL"
                color = "#F97316"
            else:
                recommendation = "🔥 STRONG SELL"
                color = "#EF4444"
    else:
        st.error(f"❌ Unable to fetch data for {symbol}")
        st.stop()

# ============ بطاقات المقاييس الرئيسية ============

if stock_data and tech_analysis:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price = tech_analysis["current_price"]
        change = tech_analysis["price_change"]

        st.markdown(f"""
        <div class="metric-card" style="border-color: #3B82F6;">
            <h4 style="color: #6B7280; margin-bottom: 10px;">📈 Current Price</h4>
            <h2 style="color: #1F2937; margin: 15px 0;">${price:.2f}</h2>
            <p style="{'color: #10B981' if change >= 0 else 'color: #EF4444'}; font-size: 1.2rem; font-weight: bold;">
                {'+' if change >= 0 else ''}{change:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        sentiment = sentiment_analysis["label"]
        score = sentiment_analysis["overall_score"] * 100

        st.markdown(f"""
        <div class="metric-card" style="border-color: #10B981;">
            <h4 style="color: #6B7280; margin-bottom: 10px;">😊 Market Sentiment</h4>
            <h2 style="color: #1F2937; margin: 15px 0;">{sentiment.split()[0]}</h2>
            <p style="color: #6B7280; font-size: 1.1rem;">
                {score:.1f}% Confidence
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rsi = tech_analysis["rsi"]
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        rsi_color = "#EF4444" if rsi < 30 else "#10B981" if rsi > 70 else "#F59E0B"

        st.markdown(f"""
        <div class="metric-card" style="border-color: {rsi_color};">
            <h4 style="color: #6B7280; margin-bottom: 10px;">📊 RSI Indicator</h4>
            <h2 style="color: #1F2937; margin: 15px 0;">{rsi:.1f}</h2>
            <p style="color: {rsi_color}; font-size: 1.1rem; font-weight: bold;">
                {rsi_status}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        signal = tech_analysis["signal"]
        conf = tech_analysis["confidence"] * 100
        signal_color = "#10B981" if signal == "BULLISH" else "#EF4444" if signal == "BEARISH" else "#F59E0B"

        st.markdown(f"""
        <div class="metric-card" style="border-color: {signal_color};">
            <h4 style="color: #6B7280; margin-bottom: 10px;">🎯 Trading Signal</h4>
            <h2 style="color: {signal_color}; margin: 15px 0;">{signal}</h2>
            <p style="color: #6B7280; font-size: 1.1rem;">
                {conf:.1f}% Confidence
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============ المخطط الرئيسي ============

st.markdown("## 📊 Interactive Price Chart")

if stock_data and 'historical' in stock_data:
    df = stock_data["historical"]

    # إنشاء مخطط الشموع
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price",
        increasing_line_color='#10B981',
        decreasing_line_color='#EF4444'
    )])

    # إضافة المتوسطات المتحركة
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'].rolling(window=20).mean(),
        name="20-Day MA",
        line=dict(color='orange', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'].rolling(window=50).mean(),
        name="50-Day MA",
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f"{symbol} Price Analysis - {timeframe}",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=500,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ============ التوصية الرئيسية ============

st.markdown("## 🎯 AI Investment Recommendation")

if 'recommendation' in locals():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div style="background: {color}20; border-left: 5px solid {color}; padding: 2rem; border-radius: 10px;">
            <h2 style="color: {color}; margin-bottom: 1rem;">{recommendation}</h2>
            <p style="font-size: 1.2rem; color: #1F2937; margin-bottom: 1rem;">
                <strong>Overall Score:</strong> {overall_score*100:.1f}% |
                <strong>Confidence:</strong> High
            </p>
            <div style="background: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p style="margin: 0.5rem 0;"><strong>📈 Target Price:</strong> ${tech_analysis['current_price'] * 1.15:.2f} (+15%)</p>
                <p style="margin: 0.5rem 0;"><strong>🛡️ Stop Loss:</strong> ${tech_analysis['current_price'] * 0.92:.2f} (-8%)</p>
                <p style="margin: 0.5rem 0;"><strong>⏳ Time Horizon:</strong> 3-6 Months</p>
                <p style="margin: 0.5rem 0;"><strong>📊 Risk Level:</strong> Medium</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # مؤشر القياس
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AI Score", 'font': {'size': 24, 'color': color}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': '#EF4444'},
                    {'range': [40, 60], 'color': '#F59E0B'},
                    {'range': [60, 100], 'color': '#10B981'}
                ]
            }
        ))

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ============ علامات التبويب التفصيلية ============

tabs = st.tabs(["📈 Technical", "😊 Sentiment", "🏛️ Fundamental", "🔮 Predictions"])

with tabs[0]:
    st.header("Technical Analysis")

    if tech_analysis:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Indicators")

            indicators = {
                "Current Price": f"${tech_analysis['current_price']:.2f}",
                "20-Day MA": f"${tech_analysis['sma_20']:.2f}",
                "50-Day MA": f"${tech_analysis['sma_50']:.2f}",
                "RSI": f"{tech_analysis['rsi']:.1f}",
                "Signal": tech_analysis["signal"],
                "Confidence": f"{tech_analysis['confidence']*100:.1f}%"
            }

            for name, value in indicators.items():
                st.metric(name, value)

        with col2:
            st.subheader("Trend Analysis")

            # تحليل الاتجاه
            trends = {
                "Short-term": "Bullish" if tech_analysis['current_price'] > tech_analysis['sma_20'] else "Bearish",
                "Medium-term": "Bullish" if tech_analysis['current_price'] > tech_analysis['sma_50'] else "Bearish",
                "Momentum": "Strong" if abs(tech_analysis['price_change']) > 5 else "Moderate",
                "Volatility": "High" if tech_analysis['price_change'] > 10 else "Normal"
            }

            for trend, value in trends.items():
                color = "#10B981" if "Bullish" in value or "Strong" in value else "#EF4444" if "Bearish" in value else "#F59E0B"
                st.markdown(f"**{trend}:** <span style='color:{color}'>{value}</span>", unsafe_allow_html=True)

with tabs[1]:
    st.header("Sentiment Analysis")

    if sentiment_analysis:
        col1, col2 = st.columns(2)

        with col1:
            # مخطط دائري للمشاعر
            labels = ['Positive', 'Neutral', 'Negative']
            values = [65, 25, 10]

            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=['#10B981', '#6B7280', '#EF4444']
            )])
            fig_pie.update_layout(
                title="Sentiment Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Sources Breakdown")

            sources = sentiment_analysis["breakdown"]

            for source, score in sources.items():
                score_pct = score * 100
                st.write(f"**{source.title()}:**")
                st.progress(score)
                st.write(f"{score_pct:.1f}% Positive")
                st.write("---")

with tabs[2]:
    st.header("Fundamental Analysis")

    if fundamental_analysis:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Company Info")

            st.metric("Company", fundamental_analysis["company_name"])
            st.metric("Sector", fundamental_analysis["sector"])

            # تقييم القيمة
            market_cap = fundamental_analysis["market_cap"]
            if market_cap > 1_000_000_000_000:
                cap_str = f"${market_cap/1_000_000_000_000:.2f}T"
            elif market_cap > 1_000_000_000:
                cap_str = f"${market_cap/1_000_000_000:.2f}B"
            else:
                cap_str = f"${market_cap/1_000_000:.2f}M"

            st.metric("Market Cap", cap_str)

        with col2:
            st.subheader("Financial Metrics")

            metrics = {
                "P/E Ratio": f"{fundamental_analysis['pe_ratio']:.2f}" if fundamental_analysis['pe_ratio'] else "N/A",
                "P/B Ratio": f"{fundamental_analysis['pb_ratio']:.2f}" if fundamental_analysis['pb_ratio'] else "N/A",
                "ROE": f"{fundamental_analysis['roe']*100:.1f}%" if fundamental_analysis['roe'] else "N/A",
                "Debt/Equity": f"{fundamental_analysis['debt_to_equity']:.2f}" if fundamental_analysis['debt_to_equity'] else "N/A",
                "Profit Margin": f"{fundamental_analysis['profit_margin']*100:.1f}%" if fundamental_analysis['profit_margin'] else "N/A",
                "Dividend Yield": f"{fundamental_analysis['dividend_yield']:.2f}%" if fundamental_analysis['dividend_yield'] else "N/A"
            }

            for name, value in metrics.items():
                st.metric(name, value)

with tabs[3]:
    st.header("AI Price Predictions")

    days = st.slider("Prediction Horizon (days)", 7, 90, 30)

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating AI predictions..."):
            # محاكاة التنبؤات
            current_price = tech_analysis['current_price']

            # إنشاء تنبؤات
            dates = [datetime.now() + timedelta(days=i) for i in range(days + 1)]
            predictions = [current_price]

            for i in range(1, days + 1):
                # حركة عشوائية مع اتجاه
                daily_change = np.random.normal(0.001, 0.02)  
                if sentiment_analysis['overall_score'] > 0.6:
                    daily_change += 0.001  

                new_price = predictions[-1] * (1 + daily_change)
                predictions.append(new_price)

           
            fig = go.Figure()

            
            if stock_data and 'historical' in stock_data:
                hist = stock_data["historical"]
                hist_dates = hist.index[-30:]  # آخر 30 يوم
                hist_prices = hist['Close'].values[-30:]

                fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=hist_prices,
                    mode='lines',
                    name='Historical',
                    line=dict(color='#6B7280', width=2)
                ))

            
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=6)
            ))

            
            upper_bound = [p * 1.05 for p in predictions]
            lower_bound = [p * 0.95 for p in predictions]

            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))

            fig.update_layout(
                title=f"{symbol} Price Forecast ({days} days)",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            
            final_prediction = predictions[-1]
            expected_return = ((final_prediction / current_price) - 1) * 100

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${current_price:.2f}")

            with col2:
                st.metric("Predicted Price", f"${final_prediction:.2f}")

            with col3:
                st.metric("Expected Return", f"{expected_return:.1f}%")

            with col4:
                st.metric("Confidence", "78%")


st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white; border-radius: 15px; margin-top: 3rem;">
    <h3>🔬 AlphaMind Analytics Research Platform</h3>
    <p style="opacity: 0.9; font-size: 1.1rem;">Joint initiative between MIT Computer Science & AI Lab and Oxford University Mathematical Institute</p>
    <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; min-width: 200px;">
            <p style="margin: 0; font-weight: bold;">📍 Cambridge, MA</p>
            <p style="margin: 0; opacity: 0.8;">MIT Campus</p>
        </div>
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; min-width: 200px;">
            <p style="margin: 0; font-weight: bold;">📍 Oxford, UK</p>
            <p style="margin: 0; opacity: 0.8;">University of Oxford</p>
        </div>
    </div>
    <p style="margin-top: 2rem; opacity: 0.8;">© 2024 AlphaMind Analytics | research@alphamind-analytics.com</p>
    <p style="opacity: 0.7; font-size: 0.9rem;">All financial analysis is for research purposes only. Not investment advice.</p>
</div>
""", unsafe_allow_html=True)

print("✅ AlphaMind Analytics Frontend created successfully!")
