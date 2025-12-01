
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="FinSight AI - Advanced Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .positive {
        color: #10B981;
        font-weight: bold;
    }
    .negative {
        color: #EF4444;
        font-weight: bold;
    }
    .neutral {
        color: #6B7280;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FinSightDashboard:
    """Interactive dashboard for FinSight AI"""

    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

    def run(self):
        """Run the dashboard"""
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=100)
            st.title("FinSight AI")
            st.markdown("---")

            # Symbol selection
            selected_symbol = st.selectbox(
                "Select Stock Symbol",
                self.symbols,
                index=0
            )

            # Analysis type
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Comprehensive", "Technical", "Sentiment", "Fundamental"],
                index=0
            )

            # Timeframe
            timeframe = st.select_slider(
                "Timeframe",
                options=["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"],
                value="6M"
            )

            # Additional settings
            st.markdown("---")
            st.subheader("Settings")
            show_advanced = st.checkbox("Show Advanced Metrics", value=True)
            auto_refresh = st.checkbox("Auto-refresh", value=True)

            if st.button("🔄 Refresh Analysis"):
                st.rerun()

        # Main content
        st.markdown('<h1 class="main-header">📈 FinSight AI Dashboard</h1>', unsafe_allow_html=True)

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self.display_metric_card(
                title="Current Price",
                value="$175.42",
                change="+2.34%",
                change_type="positive"
            )

        with col2:
            self.display_metric_card(
                title="Market Sentiment",
                value="Bullish",
                change="High Confidence",
                change_type="positive"
            )

        with col3:
            self.display_metric_card(
                title="RSI",
                value="62.5",
                change="Neutral",
                change_type="neutral"
            )

        with col4:
            self.display_metric_card(
                title="Volume",
                value="45.2M",
                change="Above Average",
                change_type="positive"
            )

        # Charts section
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price Chart")
            self.plot_price_chart(selected_symbol, timeframe)

        with col2:
            st.subheader("Sentiment Analysis")
            self.plot_sentiment_gauge()

        # Detailed analysis
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical", "😊 Sentiment", "🏛️ Fundamental", "📰 News"])

        with tab1:
            self.display_technical_analysis()

        with tab2:
            self.display_sentiment_analysis()

        with tab3:
            self.display_fundamental_analysis()

        with tab4:
            self.display_news_feed()

        # Trading signals
        st.markdown("---")
        self.display_trading_signals()

    def display_metric_card(self, title, value, change, change_type):
        """Display a metric card"""
        st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            <p class="{change_type}">{change}</p>
        </div>
        """, unsafe_allow_html=True)

    def plot_price_chart(self, symbol, timeframe):
        """Plot interactive price chart"""
        # Get data
        period_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo",
                     "6M": "6mo", "1Y": "1y", "5Y": "5y"}

        data = yf.download(symbol, period=period_map[timeframe])

        # Create chart
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'].rolling(20).mean(),
            name="SMA 20",
            line=dict(color='orange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'].rolling(50).mean(),
            name="SMA 50",
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=f"{symbol} Price Chart",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=400,
            showlegend=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_sentiment_gauge(self):
        """Plot sentiment gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=72.5,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'red'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 72.5
                }
            }
        ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def display_technical_analysis(self):
        """Display technical analysis section"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Indicators")

            indicators = {
                "RSI": {"value": 62.5, "status": "Neutral"},
                "MACD": {"value": 2.34, "status": "Bullish"},
                "Bollinger %B": {"value": 0.68, "status": "Neutral"},
                "Volume Trend": {"value": "↑", "status": "Bullish"}
            }

            for indicator, data in indicators.items():
                st.metric(label=indicator, value=data["value"], delta=data["status"])

        with col2:
            st.subheader("Trend Analysis")

            # Trend indicators
            trends = {
                "Short-term": "Bullish",
                "Medium-term": "Bullish",
                "Long-term": "Neutral",
                "Overall": "Bullish"
            }

            for trend, direction in trends.items():
                color = "green" if "Bullish" in direction else "red" if "Bearish" in direction else "gray"
                st.markdown(f"**{trest}:** <span style='color:{color}'>{direction}</span>",
                          unsafe_allow_html=True)

    def display_sentiment_analysis(self):
        """Display sentiment analysis section"""
        # Sentiment distribution
        sentiment_data = pd.DataFrame({
            'Source': ['News', 'Social', 'Analysts', 'Options'],
            'Sentiment': [0.75, 0.62, 0.80, 0.55],
            'Volume': [100, 85, 60, 40]
        })

        fig = px.bar(sentiment_data, x='Source', y='Sentiment',
                     color='Sentiment', title="Sentiment by Source",
                     color_continuous_scale=['red', 'yellow', 'green'])

        st.plotly_chart(fig, use_container_width=True)

        # Sentiment timeline
        st.subheader("Sentiment Timeline")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        timeline_data = pd.DataFrame({
            'Date': dates,
            'Sentiment': np.random.uniform(0.4, 0.8, 30)
        })

        fig_line = px.line(timeline_data, x='Date', y='Sentiment',
                          title="30-Day Sentiment Trend")
        st.plotly_chart(fig_line, use_container_width=True)

    def display_fundamental_analysis(self):
        """Display fundamental analysis section"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Financial Metrics")

            metrics = {
                "P/E Ratio": "28.5",
                "P/B Ratio": "8.2",
                "ROE": "45.2%",
                "Debt/Equity": "0.35",
                "Dividend Yield": "0.6%"
            }

            for metric, value in metrics.items():
                st.metric(label=metric, value=value)

        with col2:
            st.subheader("Analyst Consensus")

            consensus = {
                "Strong Buy": 15,
                "Buy": 8,
                "Hold": 5,
                "Sell": 2,
                "Strong Sell": 1
            }

            fig_pie = go.Figure(data=[go.Pie(
                labels=list(consensus.keys()),
                values=list(consensus.values()),
                hole=0.3,
                marker_colors=['#00FF00', '#90EE90', '#FFFF00', '#FFA500', '#FF0000']
            )])

            fig_pie.update_layout(title="Analyst Recommendations")
            st.plotly_chart(fig_pie, use_container_width=True)

    def display_
