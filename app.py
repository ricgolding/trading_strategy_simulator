import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(layout="wide")

# -----------------------
# Sidebar Controls
# -----------------------

run_button = st.sidebar.button("🚀 Run Strategy")

st.sidebar.header("🎯 Core Controls")

symbols_input = st.sidebar.text_input(
    "Ticker Symbols (comma-separated)", 
    value="SPY,QQQ", 
    help="Enter one or more ticker symbols separated by commas (e.g., SPY, QQQ, AAPL)"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]


interval = st.sidebar.selectbox(
    "Interval Stock Frequency",
    options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
    index=0,
    help="Select the interval frequency."
)

period = st.sidebar.selectbox(
    "Data Period (1m allowed)",
    options=["1d", "5d", "7d"],
    index=1,
    help="Select timeframe."
)

initial_investment = st.sidebar.number_input(
    "Initial Investment ($)", min_value=1000, value=10000, step=1000,
    help="Capital to buy in at the beginning."
)

backup_cash = st.sidebar.number_input(
    "Backup Cash ($)", min_value=0, value=10000, step=1000,
    help="Extra cash for reactive buys when price drops."
)

MIN_CASH_BUFFER = st.sidebar.number_input(
    "Minimum Cash Buffer ($)", value=2000, step=500,
    help="Reserve amount to never go below in cash."
)

st.sidebar.header("⚙️ Strategy Behavior")

enable_zscore = st.sidebar.checkbox(
    "Enable Z-score Filtering",
    value=False,
    help="Use Z-score thresholds to decide when to buy/sell."
)

Z_BUY = st.sidebar.slider("Z-score Buy Threshold", -3.0, 0.0, -1.0, 0.1)
Z_SELL = st.sidebar.slider("Z-score Sell Threshold", 0.0, 3.0, 1.0, 0.1)

MAX_POSITION_RATIO = st.sidebar.slider(
    "Max Position %", 0.0, 1.0, 0.75, 0.05,
    help="Limit position size relative to current portfolio."
)

target_return = st.sidebar.slider(
    "Target Return (%) for Liquidation",
    1.0, 100.0, 50.0, 1.0,
    help="Sell all shares when profit meets this %."
) / 100


st.sidebar.header("⚡ Market Conditions Filters")

enable_vix_filter = st.sidebar.checkbox(
    "Enable VIX Filtering", value=False,
    help="Only allow trades when VIX is within the selected range."
)

vix_min = st.sidebar.slider("Min VIX", 10, 40, 13, 1)
vix_max = st.sidebar.slider("Max VIX", 10, 40, 25, 1)


st.sidebar.header("📊  Visualization Toggles")

show_liquidations = st.sidebar.checkbox(
    "Show Liquidation Trades",
    value=False,
    help="Display liquidation when profit threshold is hit."
)

show_baseline = st.sidebar.checkbox(
    "Compare with Buy & Hold",
    value=False,
    help="Plot Buy & Hold benchmark for context."
)

show_components = st.sidebar.checkbox(
    "Show Portfolio Components",
    value=False,
    help="Display backup cash and market value (equity) as separate lines."
)

run_monte_carlo = st.sidebar.checkbox(
    "Run Monte Carlo by VIX Regime", 
    value=False,
    help="Compare strategy behavior in low vs high volatility periods"
)

st.sidebar.markdown("---")
st.sidebar.header("🏁 Exit Strategy")

enable_exit_conditions = st.sidebar.checkbox(
    "Enable Hybrid Exit Strategy",
    value=True,
    help="Sell all shares if price or portfolio value reaches a defined multiple."
)

price_exit_multiple = st.sidebar.slider(
    "Exit at Price x (Initial Buy Price)", 1.00, 2.00, 1.05, 0.01,
    help="Exit if stock price hits this multiple of the entry price."
)

portfolio_exit_multiple = st.sidebar.slider(
    "Exit at Portfolio x (Starting Value)", 1.00, 3.00, 1.2, 0.01,
    help="Exit if total portfolio hits this multiple of starting capital."
)

window = 20

# ----------------------
# Functions 
# ----------------------


@st.cache_data(show_spinner=False)
def fetch_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period)
    return df

def run_strategy_for_symbol(symbol):
    try:
        # -----------------------
        # Fetch Stock & Prep Data
        # -----------------------

        df = fetch_data(symbol, interval, period)
        vix = fetch_data("^VIX", interval, period)

        
        df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        
        df = df.between_time("09:30", "16:00")
        df = df[df.index.dayofweek < 5]
        df['Close'] = df['Close'].astype(float)
    
        # -----------------------
        # Fetch and Merge VIX
        # -----------------------
        vix = yf.download("^VIX", interval=interval, period=period)
        vix.index = pd.to_datetime(vix.index)
        
        if vix.index.tz is None:
            vix.index = vix.index.tz_localize("America/New_York")
        else:
            vix.index = vix.index.tz_convert("America/New_York")
        
        vix = vix.between_time("09:30", "16:00")
        vix = vix[vix.index.dayofweek < 5]
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        vix = vix[['Close']].rename(columns={"Close": "VIX"})
        
        # Merge with main stock data
        df = df.merge(vix, how='left', left_index=True, right_index=True)
        df['VIX'] = df['VIX'].ffill()
    
        #Calculate indicators
        df['zscore'] = (df['Close'] - df['Close'].rolling(window).mean()) / df['Close'].rolling(window).std()
        
        
        # -----------------------
        # Strategy Logic
        # -----------------------
        local_backup_cash = backup_cash
        starting_capital = initial_investment + backup_cash
        shares = 0
        buy_count = 0
        sell_count = 0
        portfolio_values = [np.nan] * (window + 1)  # pad to match DataFrame length
        equity_values = [np.nan] * (window + 1)
        cash_values = [np.nan] * (window + 1)
        trades = []
        has_liquidated = False
        
        # First Buy
        first_price = df['Close'].iloc[window]
        shares = initial_investment / first_price
        equity_value = shares * first_price
        portfolio_value = equity_value + backup_cash
        portfolio_values[window] = portfolio_value
        equity_values[window] = equity_value
        cash_values[window] = backup_cash
        entry_price = first_price
        price_target = entry_price * price_exit_multiple
        portfolio_target = starting_capital * portfolio_exit_multiple
        
    
        trades.append({
            'Time': df.index[window],
            'Action': 'Buy (Initial)',
            'Shares': shares,
            'Price': first_price,
            'Value': equity_value,
            'Cash': local_backup_cash,
            'Portfolio Value': portfolio_value
        })
        
        buy_count += 1

        # -----------------------
        # Main Strategy Loop
        # -----------------------
        for i in range(window + 1, len(df)):
            
            # Handle VIX filter — inject NaNs if skipping
            if enable_vix_filter:
                vix_now = df['VIX'].iloc[i]
                if vix_now < vix_min or vix_now > vix_max:
                    portfolio_values.append(np.nan)
                    equity_values.append(np.nan)
                    cash_values.append(np.nan)
                    continue
        
            price = df['Close'].iloc[i]
            z = df['zscore'].iloc[i]
            timestamp = df.index[i]
            equity_value = shares * price
            portfolio_value = local_backup_cash + equity_value
        
            # Track portfolio value at every step
            portfolio_values.append(portfolio_value)
            equity_values.append(equity_value)
            cash_values.append(local_backup_cash)
        
        
            # Skip everything if we've already liquidated
            if has_liquidated:
                continue
        
            # --- Hybrid Exit Strategy ---
            # If price OR portfolio value hits target thresholds, sell everything
            if enable_exit_conditions:
                if (price >= price_target) or (portfolio_value >= portfolio_target):
                    if shares > 0:
                        local_backup_cash += equity_value # convert shares to cash
                        trades.append({
                            'Time': timestamp,
                            'Action': 'Liquidate',
                            'Shares': shares,
                            'Price': price,
                            'Value': equity_value,
                            'Cash': local_backup_cash,
                            'Portfolio Value': portfolio_value,
                            'Note': 'Hybrid Exit'
                        })
                    shares = 0
                    has_liquidated = True
                    continue # skip trading logic after liquidation
        
            # Calculate current exposure limits
            position_value = shares * price
            max_position_value = portfolio_value * MAX_POSITION_RATIO
        
            # --- If using Z-score logic ---
            if enable_zscore:
                # BUY logic: when Z-score is below buy threshold (oversold)
                if z < Z_BUY:
                    drop_pct = abs(z) / 3  # scale investment based on how far we are from mean
                    investment = min(local_backup_cash * drop_pct, max_position_value - position_value)
                    if local_backup_cash - investment >= MIN_CASH_BUFFER and investment > 0:
                        bought_shares = investment / price
                        shares += bought_shares
                        local_backup_cash -= investment
                        trades.append({
                            'Time': timestamp, 'Action': 'Buy', 'Shares': bought_shares,
                            'Price': price, 'Value': investment,
                            'Cash': local_backup_cash, 'Portfolio Value': portfolio_value,
                            'Change %': f"{z:.4f} Z"
                        })
                        buy_count += 1
        
                # SELL logic: when Z-score is above sell threshold (overbought)
                elif z > Z_SELL and shares > 0:
                    climb_pct = z / 3 # scale how much to sell
                    sell_shares = min(shares, shares * climb_pct)
                    if sell_shares > 0:
                        proceeds = sell_shares * price
                        local_backup_cash += proceeds
                        shares -= sell_shares
                        trades.append({
                            'Time': timestamp, 'Action': 'Sell', 'Shares': sell_shares,
                            'Price': price, 'Value': proceeds,
                            'Cash': local_backup_cash, 'Portfolio Value': portfolio_value,
                            'Change %': f"{z:.4f} Z"
                        })
                        sell_count += 1
        
            # --- If Z-score logic is disabled, use price momentum instead ---
            else:
                prev_price = df['Close'].iloc[i - 1]
                price_change = (price - prev_price) / prev_price
        
                # BUY on downward move (buy the dip)
                if price_change < 0:
                    drop_pct = abs(price_change)
                    investment = min(local_backup_cash * drop_pct, max_position_value - position_value)
                    if investment > 0 and (local_backup_cash - investment >= MIN_CASH_BUFFER):
                        bought_shares = investment / price
                        shares += bought_shares
                        local_backup_cash -= investment
                        trades.append({
                            'Time': timestamp, 'Action': 'Buy',
                            'Shares': bought_shares, 'Price': price, 'Value': investment,
                            'Cash': local_backup_cash, 'Portfolio Value': portfolio_value,
                            'Change %': f"{price_change:.4%}"
                        })
                        buy_count += 1
        
                # SELL on upward move (profit taking)
                elif price_change > 0 and shares > 0:
                    climb_pct = price_change
                    sell_shares = min(shares, shares * climb_pct)
                    if sell_shares > 0:
                        proceeds = sell_shares * price
                        local_backup_cash += proceeds
                        shares -= sell_shares
                        trades.append({
                            'Time': timestamp, 'Action': 'Sell',
                            'Shares': sell_shares, 'Price': price, 'Value': proceeds,
                            'Cash': local_backup_cash, 'Portfolio Value': portfolio_value,
                            'Change %': f"{price_change:.4%}"
                        })
                        sell_count += 1
        
        # -----------------------
        # Final Data Prep
        # -----------------------
        
        # Assign values to DataFrame
        df['Portfolio_Value'] = pd.Series(portfolio_values, index=df.index)
        df['Equity_Value'] = pd.Series(equity_values, index=df.index)
        df['Backup_Cash'] = pd.Series(cash_values, index=df.index)
        
        # Forward fill and interpolate to smooth any gaps
        df[['Portfolio_Value', 'Equity_Value', 'Backup_Cash']] = (
            df[['Portfolio_Value', 'Equity_Value', 'Backup_Cash']]
            .ffill()
            .interpolate(method='linear')
        )
        
        # -----------------------
        # Risk metrics
        # -----------------------
        
        portfolio_series = pd.Series(portfolio_values, index=df.index)
        returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60)
        sortino_ratio = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252 * 6.5 * 60)
        drawdown = portfolio_series / portfolio_series.cummax() - 1
        max_drawdown = drawdown.min() * 100

        # Calculate Cumulative Return
        try:
            start_value = df['Portfolio_Value'].dropna().iloc[0]
            end_value = df['Portfolio_Value'].dropna().iloc[-1]
            cumulative_return = (end_value - start_value) / start_value * 100
        except:
            cumulative_return = 0
        
        # Buy & Hold logic (optional, only if baseline is toggled)
        if show_baseline:
            bh_shares = initial_investment / df['Close'].iloc[window]
            bh_series = df['Close'] * bh_shares
            bh_final = bh_series.iloc[-1]
            bh_return = (bh_final - initial_investment) / initial_investment * 100
        else:
            bh_return = None

        summary_dict = {
            "Ticker": symbol,
            "Final Value": round(portfolio_series.iloc[-1], 2),
            "Cumulative Return": cumulative_return,
            "Buy & Hold Return": bh_return if show_baseline else None,
            "Status": (
                "Beat Market" if show_baseline and cumulative_return > bh_return 
                else "Underperformed" if show_baseline 
                else "No Benchmark"
            ),
            "Buys": buy_count,
            "Sells": sell_count,
            "Sharpe": round(sharpe_ratio, 2),
            "Sortino": round(sortino_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Backup Cash": local_backup_cash,
            "Shares": shares,
            "Equity Value": shares * df['Close'].iloc[-1]
        }


        return df, trades, summary_dict

    except Exception as e:
        st.error(f"Error running strategy for {symbol}: {e}")
        return None, None, None


# Monte Carlo Stress Testing 
def monte_carlo_by_vix_plotly(symbol=symbols, simulations=1000, days=int(period[0]), seed=42):
    spy = yf.download(symbol, interval=interval, period=period)
    vix = yf.download("^VIX", interval=interval, period=period)

    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)

    df = pd.DataFrame(index=spy.index)
    df['Symbol_Close'] = spy['Close']
    df['VIX'] = vix['Close'].reindex(df.index, method='ffill')
    df = df.dropna()

    df['Returns'] = df['Symbol_Close'].pct_change().fillna(0)
    vix_threshold = df['VIX'].median()
    df['Volatility_Regime'] = np.where(df['VIX'] > vix_threshold, 'High VIX', 'Low VIX')

    low_vix_returns = df[df['Volatility_Regime'] == 'Low VIX']['Returns']
    high_vix_returns = df[df['Volatility_Regime'] == 'High VIX']['Returns']

    def run_simulations(returns, label, color):
        np.random.seed(seed)
        minutes = days * 390
        paths = np.zeros((simulations, minutes))
        for i in range(simulations):
            sampled_returns = np.random.choice(returns, size=minutes, replace=True)
            paths[i] = np.cumprod(1 + sampled_returns)

        final_values = paths[:, -1]
        avg_path = np.mean(paths, axis=0)

        var_5 = np.percentile(final_values, 5)
        cvar_5 = final_values[final_values <= var_5].mean()
        prob_loss = np.mean(final_values < 1)
        prob_gain = np.mean(final_values > 1)

        metrics = {
            "label": label,
            "avg_path": avg_path,
            "color": color,
            "Value at Risk (5%):": var_5,
            "Conditional VaR (Expected Shortfall):": cvar_5,
            "Probability of Loss": prob_loss,
            "Probability of Gain": prob_gain
        }
        return metrics, paths

    low_metrics, low_paths = run_simulations(low_vix_returns, "Low VIX", "green")
    high_metrics, high_paths = run_simulations(high_vix_returns, "High VIX", "red")

    fig = go.Figure()

    # Faded individual paths (Low VIX)
    for path in low_paths[:200]:
        fig.add_trace(go.Scatter(
            y=path,
            mode='lines',
            line=dict(color='green', width=1),
            opacity=0.05,
            showlegend=False
        ))
    
    # Faded individual paths (High VIX)
    for path in high_paths[:200]:
        fig.add_trace(go.Scatter(
            y=path,
            mode='lines',
            line=dict(color='red', width=1),
            opacity=0.05,
            showlegend=False
        ))
    
    #Average Lines
    fig.add_trace(go.Scatter(y=low_metrics["avg_path"], mode='lines', name='Low VIX', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=high_metrics["avg_path"], mode='lines', name='High VIX', line=dict(color='red')))
    fig.add_trace(go.Scatter(y=[1]*len(low_metrics["avg_path"]), mode='lines', name='Starting Value', line=dict(color='gray', dash='dash')))

    fig.update_layout(
        title=f"🎲 Monte Carlo Simulation by VIX Regime ({days}-Day Outlook)",
        xaxis_title="Minutes Ahead",
        yaxis_title="Normalized Portfolio Value",
        legend=dict(x=0.01, y=0.01),
        template="plotly_white"
    )

    return fig, low_metrics, high_metrics

# -----------------------
# Display
# -----------------------

if run_button:
    with st.spinner("🚀 Running strategy... Please wait."):
        for symbol in symbols:
            df, trades, summary = run_strategy_for_symbol(symbol)
        
            if df is None:
                continue
            # --- Create Trade DataFrame ---
            trades_df = pd.DataFrame(trades)
            
            # --- Trade Log ---
            if not trades_df.empty:
                st.title(f"📊 Trading Strategy Simulator {symbols}")
                st.subheader(f"📋 Trade Log {symbol}")
                if not show_liquidations:
                    trades_df = trades_df[~trades_df['Action'].str.contains("Liquidate")]
            
                trades_df['Shares'] = trades_df['Shares'].round(4)
                trades_df['Price'] = trades_df['Price'].round(2)
                trades_df['Cash'] = trades_df['Cash'].round(2)
                trades_df['Portfolio Value'] = trades_df['Portfolio Value'].round(2)
                st.dataframe(trades_df)
            
                # --- Price Chart with Trades ---
                st.subheader(f"📉 Price Chart with Trades {symbol}")
                df_plot = df.copy()
                df_plot['Action'] = None
                df_plot['Trade_Price'] = None
            
                for _, row in trades_df.iterrows():
                    df_plot.loc[row['Time'], 'Action'] = row['Action']
                    df_plot.loc[row['Time'], 'Trade_Price'] = row['Price']
            
                fig = px.line(df_plot, x=df_plot.index, y='Close', title=f"{symbol} Price and Trades")
                fig.update_traces(line=dict(color='gray'), connectgaps=False)
            
                # Add VIX as a secondary y-axis if VIX data is available
                if enable_vix_filter:
                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot['VIX'],
                        name="VIX",
                        yaxis="y2",
                        mode="lines",
                        line=dict(color='orange', dash='dot')
                    ))
                
                    fig.update_layout(
                        yaxis=dict(title=f"{symbol} Price"),
                        yaxis2=dict(
                            title="VIX",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
            
                
                if df_plot['Action'].str.contains("Buy", na=False).any():
                    buys = df_plot[df_plot['Action'].str.contains("Buy", na=False)]
                    fig.add_scatter(x=buys.index, y=buys['Trade_Price'], mode='markers', name='Buy',
                                    marker=dict(color='green', symbol='triangle-up'))
            
                if df_plot['Action'].str.contains("Sell", na=False).any():
                    sells = df_plot[df_plot['Action'] == "Sell"]
                    fig.add_scatter(x=sells.index, y=sells['Trade_Price'], mode='markers', name='Sell',
                                    marker=dict(color='red', symbol='triangle-down'))
            
                if show_liquidations:
                    liqs = df_plot[df_plot['Action'] == "Liquidate"]
                    fig.add_scatter(x=liqs.index, y=liqs['Trade_Price'], mode='markers', name='Liquidate',
                                    marker=dict(color='blue', symbol='x'))
            
                # Only apply rangebreaks for 5m+ intervals
                if interval not in ["1m", "2m"]:
                    fig.update_xaxes(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),
                            dict(bounds=[16, 9.5], pattern="hour")
                        ]
                    )
            
            
                st.plotly_chart(fig, use_container_width=True)
            
                # --- Portfolio Value Chart ---
                fig2 = px.line(df, x=df.index, y='Portfolio_Value', title="Portfolio Value Over Time")
                fig2.update_traces(name='Total Portfolio Value (Cash + Equity)', showlegend=True, connectgaps=False)
            
            
                if show_components:
                    fig2.add_scatter(
                        x=df.index, y=df['Equity_Value'],
                        mode='lines', name='Market Value (Equity)',
                        line=dict(color='green', dash='solid')
                    )
                    fig2.add_scatter(
                        x=df.index, y=df['Backup_Cash'],
                        mode='lines', name='Backup Cash',
                        line=dict(color='orange', dash='dash')
                    )
            
                # --- Cumulative Return Calculation ---
                try:
                    start_value = df['Portfolio_Value'].dropna().iloc[0]
                    end_value = df['Portfolio_Value'].dropna().iloc[-1]
                    cumulative_return = (end_value - start_value) / start_value * 100
                    formatted_return = f"{cumulative_return:+.2f}%"
                except:
                    formatted_return = "+0.00%"
                    cumulative_return = 0
                
                color = "green" if cumulative_return >= 0 else "red"
                st.markdown(
                    f"<h4 style='font-size:24px;'>📈 Cumulative Return: "
                    f"<span style='color:{color}'>{formatted_return}</span></h4>",
                    unsafe_allow_html=True
                )
            
            
            
                if show_baseline:
                    bh_shares = initial_investment / df['Close'].iloc[window]
                    bh_series = df['Close'] * bh_shares
                    fig2.add_scatter(
                        x=df.index, y=bh_series,
                        mode='lines', name='Buy & Hold',
                        line=dict(dash='dot', color='blue')
                    )
                
                    bh_final = bh_series.iloc[-1]
                    bh_return = (bh_final - initial_investment) / initial_investment * 100
                    bh_formatted = f"{bh_return:+.2f}%"
                    bh_color = "green" if bh_return >= 0 else "red"
                
                    st.markdown(
                        f"<h4 style='font-size:24px;'>📊 Buy & Hold Return Benchmark: "
                        f"<span style='color:{bh_color}'>{bh_formatted}</span></h4>",
                        unsafe_allow_html=True
                    )
            
            
                if interval not in ["1m", "2m"]:
                    fig2.update_xaxes(
                        rangebreaks=[
                            dict(bounds=["sat", "mon"]),
                            dict(bounds=[16, 9.5], pattern="hour")
                        ]
                    )
                
                st.subheader(f"💼 Portfolio Value {symbol}")
                fig2.update_yaxes(tickprefix="$")
                st.plotly_chart(fig2, use_container_width=True)
            
                # --- Summary ---
                st.subheader(f"📌 Strategy Summary {symbol}")
                summary = {
                    "Shares Held (Final)": round(summary['Shares'], 2),
                    "Total Buys": summary['Buys'],
                    "Total Sells": summary['Sells'],
                    "Sharpe Ratio": round(summary['Sharpe'], 2),
                    "Sortino Ratio": round(summary['Sortino'], 2),
                    "Max Drawdown (%)": round(summary['Max Drawdown (%)'], 2),
                    "Status": summary["Status"],
                    "Equity Value": f"${summary['Equity Value']:,.2f}",
                    "Backup Cash": f"${summary['Backup Cash']:,.2f}",
                    "Total Portfolio": f"${summary['Equity Value'] + summary['Backup Cash']:,.2f}"
                }
                st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))
            
                if run_monte_carlo:
                    # --- Monte Carlo Stress Test ----
                    st.markdown("## 📊 Monte Carlo Stress Test by Volatility Regime")
                    fig, low_metrics, high_metrics = monte_carlo_by_vix_plotly(symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Optionally show the risk metrics
                    st.markdown(f"### 📉 Risk Metrics by VIX Regime {symbol}")
                    st.write("**Low VIX:**")
                    st.write(f"- Value at Risk (5%): {low_metrics["Value at Risk (5%):"]:.4f}")
                    st.write(f"- Conditional VaR (Expected Shortfall): {low_metrics["Conditional VaR (Expected Shortfall):"]:.4f}")
                    st.write(f"- Probability of Loss: {low_metrics["Probability of Loss"] * 100:.2f}%")
                    st.write(f"- Probability of Gain: {low_metrics["Probability of Gain"] * 100:.2f}%")
                    
                    st.write("**High VIX:**")
                    st.write(f"- Value at Risk (5%): {high_metrics["Value at Risk (5%):"]:.4f}")
                    st.write(f"- Conditional VaR (Expected Shortfall): {high_metrics["Conditional VaR (Expected Shortfall):"]:.4f}")
                    st.write(f"- Probability of Loss: {high_metrics["Probability of Loss"] * 100:.2f}%")
                    st.write(f"- Probability of Gain: {high_metrics["Probability of Gain"] * 100:.2f}%")
            
            
            else:
                st.warning("⚠️ No trades were executed. Try adjusting your thresholds or toggles.")

