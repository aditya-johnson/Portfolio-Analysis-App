import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import scipy.stats
import ta
import io
import base64

# Page configuration
st.set_page_config(page_title="S&P 500 Stock Market Analysis and Visualization", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def load_data():
    companies = pd.read_csv('data/sp500_companies.csv')
    index_data = pd.read_csv('data/sp500_index.csv')
    stocks_data = pd.read_csv('data/sp500_stocks.csv')
    # Convert stock data to numeric, replacing any non-numeric values with NaN
    numeric_columns = stocks_data.select_dtypes(include=[np.number]).columns
    stocks_data[numeric_columns] = stocks_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return companies, index_data, stocks_data

def calculate_technical_indicators(df):
    try:
        # Ensure data is numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Drop any NaN values
        df = df.dropna(subset=['Close'])
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df['BB_upper'] = indicator_bb.bollinger_hband()
        df['BB_middle'] = indicator_bb.bollinger_mavg()
        df['BB_lower'] = indicator_bb.bollinger_lband()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'])
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df
    
    return df

def get_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Load data
companies_df, index_df, stocks_df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", 
    ["Market Overview", "Stock Analysis", "Sector Analysis", "Technical Analysis", "Custom Dashboard", "Portfolio Analysis"])

if page == "Market Overview":
    st.title("S&P 500 Market Overview")
    
    # Index Performance
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fig_index = px.line(index_df, x='Date', y='S&P500',
                          title='S&P 500 Index Performance')
        fig_index.update_layout(
            xaxis_title="Date",
            yaxis_title="S&P 500 Index Value",
            hovermode='x unified'
        )
        st.plotly_chart(fig_index, use_container_width=True)
    
    with col2:
        # Calculate key metrics
        index_df['Returns'] = index_df['S&P500'].pct_change()
        daily_return = index_df['Returns'].mean()
        volatility = index_df['Returns'].std() * np.sqrt(252)
        
        st.metric("Daily Average Return", f"{daily_return:.2%}")
        st.metric("Annual Volatility", f"{volatility:.2%}")
    
    with col3:
        st.metric("Total Companies", len(companies_df))
        total_market_cap = companies_df['Marketcap'].sum()
        st.metric("Total Market Cap", f"${total_market_cap/1e12:.2f}T")

elif page == "Stock Analysis":
    st.title("Stock Analysis")
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        [col for col in stocks_df.columns if col != 'Date']
    )
    
    # Technical Analysis Settings
    st.sidebar.subheader("Technical Analysis Settings")
    show_indicators = st.sidebar.multiselect(
        "Select Technical Indicators",
        ["SMA", "EMA", "Bollinger Bands", "RSI"],
        default=["SMA"]
    )
    
    # Get stock data
    stock_data = pd.DataFrame()
    stock_data['Date'] = pd.to_datetime(stocks_df['Date'])
    stock_data['Close'] = pd.to_numeric(stocks_df[selected_stock], errors='coerce')
    stock_data = stock_data.dropna()
    stock_data.set_index('Date', inplace=True)
    
    # Calculate indicators
    if show_indicators:
        if "SMA" in show_indicators:
            sma_period = st.sidebar.slider("SMA Period", 5, 200, 20)
            stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=sma_period)
        
        if "EMA" in show_indicators:
            ema_period = st.sidebar.slider("EMA Period", 5, 200, 20)
            stock_data['EMA'] = ta.trend.ema_indicator(stock_data['Close'], window=ema_period)
        
        if "Bollinger Bands" in show_indicators:
            bb_period = st.sidebar.slider("Bollinger Bands Period", 5, 50, 20)
            bb = ta.volatility.BollingerBands(close=stock_data['Close'], window=bb_period)
            stock_data['BB_upper'] = bb.bollinger_hband()
            stock_data['BB_middle'] = bb.bollinger_mavg()
            stock_data['BB_lower'] = bb.bollinger_lband()
        
        if "RSI" in show_indicators:
            rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
            stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=rsi_period)
    
    # Create main price chart
    st.subheader(f"{selected_stock} Price Chart")
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add technical indicators to the chart
    if show_indicators:
        if "SMA" in show_indicators and 'SMA' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA'],
                name=f'SMA ({sma_period})',
                line=dict(color='orange', dash='dash')
            ))
        
        if "EMA" in show_indicators and 'EMA' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['EMA'],
                name=f'EMA ({ema_period})',
                line=dict(color='green', dash='dash')
            ))
        
        if "Bollinger Bands" in show_indicators:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['BB_upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['BB_lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash')
            ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show RSI in separate chart if selected
    if "RSI" in show_indicators and 'RSI' in stock_data.columns:
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(
            yaxis_title="RSI",
            hovermode='x unified'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Display current metrics
    st.subheader("Current Metrics")
    col1, col2, col3 = st.columns(3)
    
    try:
        if not stock_data.empty and len(stock_data) > 0:
            current_price = stock_data['Close'].iloc[-1]
            if len(stock_data) > 1:
                price_change = stock_data['Close'].pct_change().iloc[-1]
            else:
                price_change = 0.0
            
            col1.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:.2%}"
            )
            
            if "RSI" in show_indicators and 'RSI' in stock_data.columns and not stock_data['RSI'].empty:
                current_rsi = stock_data['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    col2.metric("RSI", f"{current_rsi:.1f}")
                    
                    # RSI Signal
                    rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                    col3.metric("RSI Signal", rsi_signal)
        else:
            st.warning("No data available for the selected stock.")
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    # Technical Analysis Summary
    if show_indicators and not stock_data.empty and len(stock_data) > 0:
        st.subheader("Technical Analysis Summary")
        
        try:
            summary_text = []
            
            if "SMA" in show_indicators and 'SMA' in stock_data.columns:
                sma_current = stock_data['SMA'].iloc[-1]
                if not pd.isna(sma_current) and not pd.isna(current_price):
                    sma_signal = "Bullish" if current_price > sma_current else "Bearish"
                    summary_text.append(f"SMA Signal: {sma_signal} (Price {'above' if sma_signal == 'Bullish' else 'below'} SMA)")
            
            if "Bollinger Bands" in show_indicators and 'BB_upper' in stock_data.columns and 'BB_lower' in stock_data.columns:
                bb_upper = stock_data['BB_upper'].iloc[-1]
                bb_lower = stock_data['BB_lower'].iloc[-1]
                if not pd.isna(bb_upper) and not pd.isna(bb_lower) and not pd.isna(current_price):
                    if current_price > bb_upper:
                        bb_signal = "Overbought"
                    elif current_price < bb_lower:
                        bb_signal = "Oversold"
                    else:
                        bb_signal = "Neutral"
                    summary_text.append(f"Bollinger Bands Signal: {bb_signal}")
            
            if summary_text:
                for text in summary_text:
                    st.write(text)
        except Exception as e:
            st.error(f"Error generating technical analysis summary: {str(e)}")

elif page == "Sector Analysis":
    st.title("Sector Analysis")
    
    # Sector selection
    selected_sector = st.selectbox("Select Sector", companies_df['Sector'].unique())
    sector_companies = companies_df[companies_df['Sector'] == selected_sector].copy()
    
    # Sort companies by market cap and get top 10
    sector_companies['MarketCapB'] = sector_companies['Marketcap'] / 1e9  # Convert to billions
    sector_companies = sector_companies.sort_values('MarketCapB', ascending=False)
    
    # Calculate 'Others' category if more than 10 companies
    if len(sector_companies) > 10:
        top_10_companies = sector_companies.iloc[:10]
        others_marketcap = sector_companies.iloc[10:]['MarketCapB'].sum()
        others_row = pd.DataFrame({
            'Symbol': ['Others'],
            'Shortname': ['Other Companies'],
            'MarketCapB': [others_marketcap],
            'Marketcap': [others_marketcap * 1e9]
        })
        plot_data = pd.concat([top_10_companies, others_row])
    else:
        plot_data = sector_companies
    
    # Sector metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Create market cap distribution pie chart with improved styling
        fig_market_cap = px.pie(
            plot_data, 
            values='MarketCapB',
            names='Symbol',
            title=f'Market Cap Distribution - {selected_sector}',
            hover_data={
                'Symbol': True,
                'Shortname': True,
                'MarketCapB': ':.2f'
            },
            custom_data=['Shortname', 'MarketCapB']
        )
        
        # Update layout and traces
        fig_market_cap.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Symbol: %{label}<br>" +
                         "Market Cap: $%{customdata[1]:.2f}B<br>" +
                         "Percentage: %{percent}<br><extra></extra>",
            rotation=90
        )
        
        fig_market_cap.update_layout(
            showlegend=False,
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            margin=dict(t=80, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig_market_cap, use_container_width=True)
    
    with col2:
        st.metric("Number of Companies", len(sector_companies))
        sector_market_cap = sector_companies['Marketcap'].sum()
        st.metric("Total Sector Market Cap", f"${sector_market_cap/1e9:.2f}B")
        
        # Calculate sector percentage of total market
        total_market_cap = companies_df['Marketcap'].sum()
        sector_percentage = (sector_market_cap / total_market_cap) * 100
        st.metric("% of S&P 500", f"{sector_percentage:.1f}%")
        
        # Add top companies table
        st.subheader("Top Companies by Market Cap")
        top_5_df = sector_companies.head().copy()
        top_5_df['Market Cap ($B)'] = top_5_df['MarketCapB'].round(2)
        st.dataframe(
            top_5_df[['Symbol', 'Shortname', 'Market Cap ($B)']].set_index('Symbol'),
            hide_index=False,
            use_container_width=True
        )

elif page == "Technical Analysis":
    st.title("Advanced Technical Analysis")
    
    # Stock selector
    available_stocks = [col for col in stocks_df.columns if col != 'Date']
    selected_stock = st.selectbox(
        "Select Stock",
        available_stocks,
        format_func=lambda x: f"{x}"
    )
    
    try:
        # Get stock data
        stock_data = stocks_df[['Date', selected_stock]].copy()
        stock_data.columns = ['Date', 'Close']
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
        stock_data = stock_data.dropna(subset=['Close'])
        stock_data.set_index('Date', inplace=True)
        
        # Technical parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("SMA Window")
            sma_window = st.slider("", 5, 200, 20, key='sma')
        with col2:
            st.subheader("Bollinger Bands Window")
            bb_window = st.slider("", 5, 50, 20, key='bb')
        with col3:
            st.subheader("RSI Window")
            rsi_window = st.slider("", 5, 50, 14, key='rsi')
        
        # Calculate indicators with custom parameters
        stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=sma_window)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=stock_data['Close'], window=bb_window, window_dev=2)
        stock_data['BB_upper'] = bb.bollinger_hband()
        stock_data['BB_middle'] = bb.bollinger_mavg()
        stock_data['BB_lower'] = bb.bollinger_lband()
        
        # RSI
        stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=rsi_window)
        
        # Display charts
        st.subheader("Price and Indicators")
        fig = go.Figure()
        
        # Price and SMA
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            name='Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['SMA'],
            name=f'SMA ({sma_window})',
            line=dict(color='orange', dash='dash')
        ))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{selected_stock} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI Chart
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(
            yaxis_title="RSI",
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Technical Analysis Summary
        st.subheader("Technical Analysis Summary")
        
        try:
            summary_text = []
            
            # Get current values
            current_price = stock_data['Close'].iloc[-1]
            sma_current = stock_data['SMA'].iloc[-1]
            rsi_current = stock_data['RSI'].iloc[-1]
            bb_upper = stock_data['BB_upper'].iloc[-1]
            bb_lower = stock_data['BB_lower'].iloc[-1]
            
            # Display current metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric(f"SMA ({sma_window})", f"${sma_current:.2f}")
                st.metric("RSI", f"{rsi_current:.1f}")
            
            with col2:
                st.metric("BB Upper Band", f"${bb_upper:.2f}")
                st.metric("BB Lower Band", f"${bb_lower:.2f}")
                
                # Trend signals
                if current_price > sma_current:
                    trend = "Bullish ↑"
                    trend_color = "green"
                else:
                    trend = "Bearish ↓"
                    trend_color = "red"
                st.markdown(f"**Trend Signal:** <span style='color:{trend_color}'>{trend}</span>", unsafe_allow_html=True)
            
            # Generate signals
            signals = []
            
            # SMA Signal
            sma_signal = "Bullish" if current_price > sma_current else "Bearish"
            signals.append(f"SMA Signal: {sma_signal} (Price {'above' if sma_signal == 'Bullish' else 'below'} SMA)")
            
            # Bollinger Bands Signal
            if current_price > bb_upper:
                bb_signal = "Overbought"
            elif current_price < bb_lower:
                bb_signal = "Oversold"
            else:
                bb_signal = "Neutral"
            signals.append(f"Bollinger Bands Signal: {bb_signal}")
            
            # RSI Signal
            if rsi_current > 70:
                rsi_signal = "Overbought"
            elif rsi_current < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
            signals.append(f"RSI Signal: {rsi_signal}")
            
            # Display signals
            st.subheader("Technical Signals")
            for signal in signals:
                st.write(signal)
            
            # Download data option
            if st.button('Download Analysis Data'):
                st.markdown(get_download_link(stock_data, f"{selected_stock}_technical_analysis.csv"),
                           unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating technical analysis summary: {str(e)}")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

elif page == "Custom Dashboard":
    st.title("Custom Dashboard")
    
    # Dashboard settings
    st.sidebar.subheader("Dashboard Settings")
    
    # Stock selection
    available_stocks = [col for col in stocks_df.columns if col != 'Date']
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks to Monitor",
        available_stocks,
        default=available_stocks[:3] if len(available_stocks) > 3 else available_stocks
    )
    
    # Metrics selection
    st.sidebar.subheader("Select Metrics")
    show_price = st.sidebar.checkbox("Price Chart", value=True)
    show_volume = st.sidebar.checkbox("Volume", value=False)
    show_indicators = st.sidebar.checkbox("Technical Indicators", value=True)
    
    if show_indicators:
        indicator_options = st.sidebar.multiselect(
            "Select Indicators",
            ["Moving Averages", "Bollinger Bands", "RSI"],
            default=["Moving Averages"]
        )
    
    # Time range selection
    st.sidebar.subheader("Time Range")
    date_range = st.sidebar.selectbox(
        "Select Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
        index=2
    )
    
    if selected_stocks:
        # Create dashboard layout
        for stock in selected_stocks:
            st.subheader(f"{stock} Analysis")
            
            try:
                # Get stock data
                stock_data = stocks_df[['Date', stock]].copy()
                stock_data.columns = ['Date', 'Close']
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
                stock_data = stock_data.dropna(subset=['Close'])
                stock_data.set_index('Date', inplace=True)
                
                # Filter data based on selected time range
                if date_range != "All Time":
                    months = int(date_range.split()[0])
                    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months)
                    stock_data = stock_data[stock_data.index > cutoff_date]
                
                # Calculate indicators if selected
                if show_indicators:
                    if "Moving Averages" in indicator_options:
                        stock_data['SMA_20'] = ta.trend.sma_indicator(stock_data['Close'], window=20)
                        stock_data['SMA_50'] = ta.trend.sma_indicator(stock_data['Close'], window=50)
                    
                    if "Bollinger Bands" in indicator_options:
                        bb = ta.volatility.BollingerBands(close=stock_data['Close'], window=20)
                        stock_data['BB_upper'] = bb.bollinger_hband()
                        stock_data['BB_lower'] = bb.bollinger_lband()
                    
                    if "RSI" in indicator_options:
                        stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'])
                
                # Create charts
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Price chart
                    if show_price:
                        fig = go.Figure()
                        
                        # Add price line
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            name='Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add indicators
                        if show_indicators:
                            if "Moving Averages" in indicator_options:
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['SMA_20'],
                                    name='SMA 20',
                                    line=dict(color='orange', dash='dash')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['SMA_50'],
                                    name='SMA 50',
                                    line=dict(color='green', dash='dash')
                                ))
                            
                            if "Bollinger Bands" in indicator_options:
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['BB_upper'],
                                    name='BB Upper',
                                    line=dict(color='gray', dash='dash')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index,
                                    y=stock_data['BB_lower'],
                                    name='BB Lower',
                                    line=dict(color='gray', dash='dash')
                                ))
                        
                        fig.update_layout(
                            title=f"{stock} Price Movement",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # RSI chart if selected
                        if show_indicators and "RSI" in indicator_options:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(
                                x=stock_data.index,
                                y=stock_data['RSI'],
                                name='RSI',
                                line=dict(color='purple')
                            ))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                            fig_rsi.update_layout(
                                title="RSI",
                                yaxis_title="RSI",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # Summary metrics
                    current_price = stock_data['Close'].iloc[-1]
                    price_change = stock_data['Close'].pct_change().iloc[-1]
                    
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:.2%}"
                    )
                    
                    if show_indicators:
                        if "Moving Averages" in indicator_options:
                            sma20 = stock_data['SMA_20'].iloc[-1]
                            sma50 = stock_data['SMA_50'].iloc[-1]
                            st.metric("SMA 20", f"${sma20:.2f}")
                            st.metric("SMA 50", f"${sma50:.2f}")
                        
                        if "RSI" in indicator_options:
                            rsi = stock_data['RSI'].iloc[-1]
                            st.metric("RSI", f"{rsi:.1f}")
                    
                    # Trend signal
                    if show_indicators and "Moving Averages" in indicator_options:
                        if current_price > sma20:
                            trend = "Bullish ↑"
                            trend_color = "green"
                        else:
                            trend = "Bearish ↓"
                            trend_color = "red"
                        st.markdown(f"**Trend Signal:** <span style='color:{trend_color}'>{trend}</span>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing {stock}: {str(e)}")
            
            st.markdown("---")  # Add separator between stocks
    
    else:
        st.info("Please select at least one stock to display in the dashboard.")

elif page == "Portfolio Analysis":
    st.title("Portfolio Analysis")
    
    # Stock selection
    available_stocks = [col for col in stocks_df.columns if col != 'Date']
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Portfolio",
        available_stocks,
        default=available_stocks[:4] if len(available_stocks) >= 4 else available_stocks
    )
    
    if selected_stocks:
        num_stocks = len(selected_stocks)
        
        # Initialize equal weights
        default_weight = 1.0 / num_stocks
        
        # Create individual weight sliders for each stock
        weights = []
        remaining_weight = 1.0
        
        st.sidebar.subheader("Portfolio Weights")
        for i, stock in enumerate(selected_stocks[:-1]):  # All except last stock
            weight = st.sidebar.slider(
                f"{stock} Weight",
                0.0,
                remaining_weight,
                default_weight,
                0.01
            )
            weights.append(weight)
            remaining_weight -= weight
        
        # Last stock gets the remaining weight
        weights.append(round(remaining_weight, 2))
        st.sidebar.text(f"{selected_stocks[-1]} Weight: {weights[-1]:.2f}")
        
        # Normalize weights to ensure they sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Display current portfolio allocation
        st.subheader("Current Portfolio Allocation")
        allocation_data = pd.DataFrame({
            'Stock': selected_stocks,
            'Weight': weights
        })
        
        # Create a pie chart of portfolio allocation
        fig = px.pie(allocation_data, values='Weight', names='Stock', 
                    title='Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
        
        try:
            # Calculate portfolio metrics with reduced data size
            # Resample data to weekly frequency to reduce size
            portfolio_data = pd.DataFrame()
            portfolio_data['Date'] = pd.to_datetime(stocks_df['Date'])
            portfolio_data.set_index('Date', inplace=True)
            
            # Calculate portfolio returns
            returns_df = pd.DataFrame()
            for stock, weight in zip(selected_stocks, weights):
                stock_data = pd.to_numeric(stocks_df[stock], errors='coerce')
                returns_df[stock] = stock_data.pct_change()
            
            # Resample to weekly data
            returns_df.index = pd.to_datetime(stocks_df['Date'])
            weekly_returns = returns_df.resample('W').mean()
            
            portfolio_returns = (weekly_returns * weights).sum(axis=1)
            portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Portfolio Performance Chart
            st.subheader("Portfolio Performance")
            fig = go.Figure()
            
            # Add portfolio performance line
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative_returns.index,
                y=portfolio_cumulative_returns,
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            # Add individual stock performance with reduced data
            for stock, weight in zip(selected_stocks, weights):
                stock_data = pd.to_numeric(stocks_df[stock], errors='coerce')
                stock_returns = stock_data.pct_change()
                stock_returns.index = pd.to_datetime(stocks_df['Date'])
                weekly_stock_returns = stock_returns.resample('W').mean()
                stock_cumulative_returns = (1 + weekly_stock_returns).cumprod()
                
                fig.add_trace(go.Scatter(
                    x=stock_cumulative_returns.index,
                    y=stock_cumulative_returns,
                    name=stock,
                    line=dict(dash='dash', width=1)
                ))
            
            fig.update_layout(
                title="Weekly Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display portfolio metrics
            st.subheader("Portfolio Metrics")
            col1, col2, col3 = st.columns(3)
            
            # Annualized Return (based on weekly data)
            annual_return = portfolio_returns.mean() * 52
            col1.metric("Annual Return", f"{annual_return:.2%}")
            
            # Annualized Volatility (based on weekly data)
            annual_vol = portfolio_returns.std() * np.sqrt(52)
            col2.metric("Annual Volatility", f"{annual_vol:.2%}")
            
            # Sharpe Ratio (assuming risk-free rate of 0.02)
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
            col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Correlation Matrix with reduced size
            st.subheader("Correlation Matrix")
            # Use weekly returns for correlation matrix
            corr_matrix = weekly_returns.corr()
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
        
    else:
        st.info("Please select at least one stock for portfolio analysis.")