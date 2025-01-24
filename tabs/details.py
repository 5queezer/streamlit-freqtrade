import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from charts.allcharts import (
    draw_drawdown_chart,
    draw_winloss_ratio_chart,
    draw_distribution_charts,
    draw_cumulative_profit_chart,
)
from utils.trades import extract_trades


def show_details(query_params: dict):
    """Displays strategy details and KPIs in an optimized format with icons and colorized table."""
    strategy_name = query_params["strategy"]
    json_path = query_params["json_path"]
    trades_df, results, error = extract_trades(json_path)

    if error:
        st.error(error)
        return

    st.title(f"Details for {strategy_name}")

    # --- Extract Balance Information ---
    start_balance = results["Start Balance"]
    end_balance = results["End Balance"]
    balance_diff = end_balance - start_balance
    balance_color = "🟢" if balance_diff >= 0 else "🔴"

    # --- KPI Display ---
    st.subheader("📊 Backtest Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Start Date", results["Start Date"])
    col1.metric("📅 End Date", results["End Date"])
    col1.metric("📆 Days", results["Days"])

    col2.metric("💰 Profit %", f"{results['Profit %']}%")
    col2.metric("📈 CAGR", f"{results['CAGR']}" if results["CAGR"] != "N/A" else "-")
    col2.metric("📉 Max Drawdown", f"{results['Max Drawdown']}%" if results["Max Drawdown"] != "N/A" else "-")

    col3.metric("📊 Sortino Ratio", results["Sortino"] if results["Sortino"] != "N/A" else "-")
    col3.metric("📊 Sharpe Ratio", results["Sharpe"] if results["Sharpe"] != "N/A" else "-")
    col3.metric("⚖️ Calmar Ratio", results["Calmar Ratio"] if results["Calmar Ratio"] != "N/A" else "-")

    st.divider()

    # --- Balance KPIs ---
    st.subheader("💰 Account Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏦 Starting Balance", f"${start_balance:,.2f}")
    col2.metric("💸 Ending Balance", f"${end_balance:,.2f}")
    col3.metric(f"📊 Difference", f"{balance_color} ${balance_diff:,.2f}")

    st.divider()

    # --- Trade Data Table ---
    st.subheader("📜 Trade Data")

    # Ensure numeric values
    trades_df["profit_abs"] = trades_df["profit_abs"].astype(float)

    # Calculate profit percentage and add color
    max_profit = trades_df["profit_abs"].abs().max()
    trades_df["profit_perc"] = (trades_df["profit_abs"] / max_profit) * 100 if max_profit != 0 else 0
    trades_df["profit_color"] = trades_df["profit_abs"].apply(lambda x: "🟢" if x >= 0 else "🔴")

    # Calculate trade duration
    trades_df["trade_duration"] = trades_df["close_date"] - trades_df["open_date"]
    trades_df["duration_display"] = trades_df["trade_duration"].apply(
        lambda x: f"{x.days}d" if x.days > 0 else f"{x.seconds // 3600}:{(x.seconds // 60) % 60:02d}"
    )

    # Select relevant columns and copy to prevent modifying original DataFrame
    trade_table = trades_df[["pair", "profit_abs", "profit_perc", "profit_color", "duration_display"]].copy()

    # Rename columns
    trade_table.rename(
        columns={"profit_abs": "Profit (Fiat)", "profit_perc": "Profit %", "duration_display": "Duration"},
        inplace=True)

    # Apply color formatting
    trade_table["Profit (Fiat)"] = trade_table.apply(
        lambda row: f"{row['profit_color']} {row['Profit (Fiat)']:,.2f}", axis=1
    )
    trade_table["Profit %"] = trade_table.apply(
        lambda row: f"{row['profit_color']} {row['Profit %']:,.2f}%", axis=1
    )

    # Display Data Table
    st.dataframe(trade_table[["pair", "Profit (Fiat)", "Profit %", "Duration"]])

    # --- Charts ---
    st.subheader("📈 Profit Over Time")
    fig_cp = draw_cumulative_profit_chart(trades_df, strategy_name)
    st.plotly_chart(fig_cp, use_container_width=True)

    st.subheader("📉 Drawdown Chart")
    fig_dd = draw_drawdown_chart(trades_df, strategy_name)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("📊 Win/Loss Ratio")
    fig_wl = draw_winloss_ratio_chart(trades_df, strategy_name)
    st.plotly_chart(fig_wl, use_container_width=True)

    # --- Cumulative Profit & Win/Draw/Loss Counts ---
    st.subheader("📊 Cumulative Profit & Win/Draw/Loss Counts")

    df_sorted = trades_df.sort_values("open_date").copy()
    df_sorted["cumulative_wins"] = (df_sorted["profit_abs"] > 0).cumsum()
    df_sorted["cumulative_draws"] = (df_sorted["profit_abs"] == 0).cumsum()
    df_sorted["cumulative_losses"] = (df_sorted["profit_abs"] < 0).cumsum()
    df_sorted["cumulative_profit"] = df_sorted["profit_abs"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_sorted["open_date"], y=df_sorted["cumulative_wins"], name="Cumulative Wins",
                             line=dict(color="green")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sorted["open_date"], y=df_sorted["cumulative_draws"], name="Cumulative Draws",
                             line=dict(color="blue")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sorted["open_date"], y=df_sorted["cumulative_losses"], name="Cumulative Losses",
                             line=dict(color="red")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sorted["open_date"], y=df_sorted["cumulative_profit"], name="Cumulative Profit",
                             line=dict(color="gold")), secondary_y=True)

    fig.update_xaxes(type="date", title_text="Time")
    fig.update_yaxes(title_text="Counts", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Profit", secondary_y=True)
    fig.update_layout(
        title=f"Strategy {strategy_name}: Win/Draw/Lose Counts & Profit",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribution Charts ---
    st.subheader("📊 Winrate & Profit Distributions")
    fig_dist = draw_distribution_charts(trades_df, strategy_name)
    st.plotly_chart(fig_dist, use_container_width=True)
