import glob
import json
import math
import os
import zipfile
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="Freqtrade Multi-Backtest Analyzer")

# Remove scrollbars and expand tables
st.markdown(
    """
    <style>
    .block-container .dataframe-container {
        max-height: none !important;
        overflow-y: visible !important;
    }
    thead tr th {
        background-color: #1f2f3f !important;
        color: #ffffff !important;
        text-align: center !important;
    }
    tbody tr td {
        background-color: #2f3f4f !important;
        color: #eeeeee !important;
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(filepath: str):
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    with z.open(name) as f:
                        return json.load(f)
        return None
    else:
        with open(filepath, "r") as f:
            return json.load(f)


def compute_streaks(trades_df: pd.DataFrame) -> dict:
    df_sorted = trades_df.sort_values("open_date")
    outcomes = (df_sorted["profit_abs"] > 0).astype(int).tolist()

    win_count, loss_count = 0, 0
    max_win, max_loss = 0, 0
    win_streaks, loss_streaks = [], []

    for outcome in outcomes:
        if outcome == 1:
            win_count += 1
            max_win = max(max_win, win_count)
            if loss_count > 0:
                loss_streaks.append(loss_count)
            loss_count = 0
        else:
            loss_count += 1
            max_loss = max(max_loss, loss_count)
            if win_count > 0:
                win_streaks.append(win_count)
            win_count = 0

    if win_count > 0:
        win_streaks.append(win_count)
    if loss_count > 0:
        loss_streaks.append(loss_count)

    avg_win = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

    return {
        "Winstr. Max": max_win,
        "Winstr. Avg": round(avg_win, 1),
        "Losestr. Max": max_loss,
        "Losestr. Avg": round(avg_loss, 1),
    }


def find_backtest_files(folder: str):
    meta_files = glob.glob(os.path.join(folder, "*.meta.json"))
    data_files = glob.glob(os.path.join(folder, "*[!meta].json")) + glob.glob(os.path.join(folder, "*.zip"))
    file_map = {os.path.basename(f): f for f in data_files}
    backtest_data = []

    for meta_path in meta_files:
        meta_data = load_json(meta_path)
        if not meta_data:
            continue
        strategy_name = list(meta_data.keys())[0]
        guessed_json = os.path.basename(meta_path).replace(".meta.json", ".json")
        json_path = file_map.get(guessed_json)
        if not json_path:
            guessed_zip = guessed_json.replace(".json", ".zip")
            json_path = file_map.get(guessed_zip)
        if json_path and os.path.exists(json_path):
            backtest_data.append((strategy_name, meta_data[strategy_name], json_path))
    return backtest_data


def extract_trades(json_path: str):
    data = load_json(json_path)
    if not data or "strategy" not in data:
        return None, None, "Invalid data."
    strategy_keys = list(data["strategy"].keys())
    if not strategy_keys:
        return None, None, "Invalid strategy key."

    strategy_name = strategy_keys[0]
    strategy_data = data["strategy"].get(strategy_name, {})
    trades = strategy_data.get("trades", [])
    if not trades:
        return None, None, "No trades found."

    trades_df = pd.DataFrame(trades)
    trades_df["profit_abs"] = trades_df["profit_abs"].astype(float)
    trades_df["open_date"] = pd.to_datetime(trades_df["open_date"])
    trades_df["close_date"] = pd.to_datetime(trades_df["close_date"])

    start_date = trades_df["open_date"].min()
    end_date = trades_df["close_date"].max()
    days_diff = (end_date - start_date).days + 1 if pd.notnull(end_date) else 0

    start_balance = 10000.0
    total_profit = trades_df["profit_abs"].sum()
    end_balance = start_balance + total_profit
    profit_pct = ((end_balance - start_balance) / start_balance) * 100 if start_balance else 0

    max_dd = strategy_data.get("max_drawdown", float("nan"))
    cagr = strategy_data.get("cagr", float("nan"))
    sortino = strategy_data.get("sortino", float("nan"))
    sharpe = strategy_data.get("sharpe", float("nan"))

    try:
        if (not math.isnan(cagr)) and (not math.isnan(max_dd)) and max_dd != 0:
            calmar = cagr / max_dd
        else:
            calmar = float("nan")
    except:
        calmar = float("nan")

    pos_sum = trades_df.loc[trades_df["profit_abs"] > 0, "profit_abs"].sum()
    neg_sum = trades_df.loc[trades_df["profit_abs"] < 0, "profit_abs"].sum()
    profit_factor = pos_sum / abs(neg_sum) if neg_sum else float("nan")

    pair_counts = trades_df["pair"].nunique()
    pairs_pct = (pair_counts / 50) * 100

    streaks = compute_streaks(trades_df)
    win_rate = (trades_df["profit_abs"] > 0).mean() * 100

    score = (
            (profit_pct * 0.2)
            + (win_rate * 0.2)
            + ((sortino if not math.isnan(sortino) else 0) * 0.2)
            + ((sharpe if not math.isnan(sharpe) else 0) * 0.2)
            - ((max_dd if not math.isnan(max_dd) else 0) * 0.2)
    )

    results = {
        "Strategy": strategy_name,
        "Timeframe": strategy_data.get("timeframe", "N/A"),
        "End Balance": round(end_balance, 2),
        "Profit %": round(profit_pct, 2),
        "Win %": round(win_rate, 2),
        "Trades": len(trades_df),
        "Winstr. Max": streaks["Winstr. Max"],
        "Winstr. Avg": streaks["Winstr. Avg"],
        "Losestr. Max": streaks["Losestr. Max"],
        "Losestr. Avg": streaks["Losestr. Avg"],
        "CAGR": round(cagr, 2) if not math.isnan(cagr) else "N/A",
        "Max Drawdown": round(max_dd, 2) if not math.isnan(max_dd) else "N/A",
        "Calmar Ratio": round(calmar, 2) if not math.isnan(calmar) else "N/A",
        "Sortino": round(sortino, 2) if not math.isnan(sortino) else "N/A",
        "Sharpe": round(sharpe, 2) if not math.isnan(sharpe) else "N/A",
        "Profit Factor": round(profit_factor, 2) if not math.isnan(profit_factor) else "N/A",
        "Pairs %": round(pairs_pct, 2),
        "Total Score": round(score, 1),
        "Start Date": str(start_date.date()) if pd.notnull(start_date) else "N/A",
        "End Date": str(end_date.date()) if pd.notnull(end_date) else "N/A",
        "Days": days_diff,
    }
    return trades_df, results, None


def draw_drawdown_chart(df: pd.DataFrame, strategy_name: str):
    """
    Plots a drawdown chart over time using the running peak logic.
    Adds a horizontal line for the average drawdown,
    plus a dashed vertical line at a datetime bull_market_top.
    """
    df_sorted = df.sort_values("open_date").copy()
    df_sorted["cumulative_profit"] = df_sorted["profit_abs"].cumsum()
    df_sorted["peak"] = df_sorted["cumulative_profit"].cummax()
    df_sorted["drawdown"] = (df_sorted["peak"] - df_sorted["cumulative_profit"]) / df_sorted["peak"] * 100
    df_sorted["drawdown"] = df_sorted["drawdown"].fillna(0)

    avg_dd = df_sorted["drawdown"].mean()
    # This time as a datetime so that the axis is consistent
    bull_market_top = datetime(2021, 11, 10).timestamp()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_sorted["open_date"], y=df_sorted["drawdown"], name="Drawdown (%)",
            line=dict(color="red")
        )
    )
    fig.add_hline(
        y=avg_dd,
        line=dict(color="green"),
        annotation_text=f"Average Drawdown (%): {round(avg_dd, 2)}",
        annotation_position="top left"
    )
    # Since the x-axis is datetime, we pass a real datetime object
    fig.add_vline(
        x=bull_market_top,
        line=dict(color="yellow", dash="dash"),
        annotation_text="2021 Bull market top",
        annotation_position="top right"
    )
    fig.update_xaxes(type="date")
    fig.update_layout(
        title=f"Strategy {strategy_name} Drawdown Analysis",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="x unified",
    )
    return fig


def draw_winloss_ratio_chart(df: pd.DataFrame, strategy_name: str):
    """
    Plots the Win/Loss ratio by week. Ratio >= 1 in green bars, < 1 in red bars,
    plus a vline for the 2021 bull market top as a real datetime object.
    """
    df_sorted = df.sort_values("open_date").copy()
    df_sorted.set_index("open_date", inplace=True)

    # Resample weekly
    weekly = df_sorted.resample("W").agg(
        profit_abs=lambda x: (x > 0).sum() - (x < 0).sum()
    )
    weekly["wins"] = df_sorted["profit_abs"].resample("W").apply(lambda x: (x > 0).sum())
    weekly["losses"] = df_sorted["profit_abs"].resample("W").apply(lambda x: (x < 0).sum())

    def ratio_func(row):
        if row["losses"] == 0 and row["wins"] > 0:
            return float("inf")
        if row["losses"] == 0 and row["wins"] == 0:
            return 0
        return row["wins"] / row["losses"]

    weekly["ratio"] = weekly.apply(ratio_func, axis=1)

    weekly["ratio_pos"] = weekly["ratio"].where(weekly["ratio"] >= 1)
    weekly["ratio_neg"] = weekly["ratio"].where(weekly["ratio"] < 1)

    bull_market_top = datetime(2021, 11, 10).timestamp()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=weekly.index, y=weekly["ratio_pos"], name="Ratio >= 1", marker_color="green")
    )
    fig.add_trace(
        go.Bar(x=weekly.index, y=weekly["ratio_neg"], name="Ratio < 1", marker_color="red")
    )
    fig.add_vline(
        x=bull_market_top,
        line=dict(color="yellow", dash="dash"),
        annotation_text="2021 Bull market top",
        annotation_position="top right"
    )
    fig.update_xaxes(type="date")
    fig.update_layout(
        barmode="relative",
        title=f"Strategy {strategy_name}: Win/Loss Ratio by Week",
        xaxis_title="Time",
        yaxis_title="Win/Loss Ratio",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="x unified",
    )
    return fig


def draw_distribution_charts(df: pd.DataFrame):
    """
    Two side-by-side boxplots:
    1) Winrate distribution by pair
    2) Profit distribution by pair
    """
    by_pair = df.groupby("pair")
    winrates = by_pair.apply(lambda x: (x["profit_abs"] > 0).mean())
    profits = by_pair["profit_abs"].sum()

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Winrate Distribution", "Profit Distribution"])

    fig.add_trace(
        go.Box(y=winrates, name="Winrate", boxpoints="all", jitter=0.3, marker_color="#95DAC1"),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=profits, name="Profit", boxpoints="all", jitter=0.3, marker_color="#95A7DA"),
        row=1, col=2
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    fig.update_layout(
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="closest",
        width=1200,
        height=600,
    )
    return fig


def compare_strategies(all_strategies):
    """
    Renders a multi-line chart comparing cumulative profits
    for each strategy's trades, grouped weekly.
    Adds a vertical line for the 2021 bull market top as a datetime.
    """
    bull_market_top = datetime(2021, 11, 10).timestamp()
    fig = go.Figure()

    for strat_name, df in all_strategies.items():
        if df.empty:
            continue
        df_sorted = df.sort_values("open_date").copy()
        df_sorted.set_index("open_date", inplace=True)

        weekly = df_sorted["profit_abs"].resample("W").sum()
        cum_profit = weekly.cumsum()

        fig.add_trace(
            go.Scatter(
                x=cum_profit.index, y=cum_profit,
                mode="lines", name=strat_name
            )
        )

    fig.add_vline(
        x=bull_market_top,
        line=dict(color="yellow", dash="dash"),
        annotation_text="2021 Bull market top",
        annotation_position="top right"
    )
    fig.update_xaxes(type="date")
    fig.update_layout(
        title="Top strategies cumulative profits performance",
        xaxis_title="Time",
        yaxis_title="Cumulative Profit",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="x unified",
    )
    return fig


def main():
    query_params = st.query_params
    in_details = ("strategy" in query_params and "json_path" in query_params)

    if not in_details:
        st.title("Freqtrade Multi-Backtest Analyzer")
        folder_dir = os.environ.get("BACKTEST_FOLDER", "./backtests")
        folder = st.text_input("Enter folder path containing backtest results:", folder_dir)

        if st.button("Scan Folder"):
            if not os.path.exists(folder):
                st.error("Folder not found.")
                return
            pbar = st.progress(0, text="Scanning...")
            backtest_files = find_backtest_files(folder)
            if not backtest_files:
                st.warning("No valid backtest files found.")
                return

            all_results = []
            total_count = len(backtest_files)
            all_trades_data = {}  # For multi-strategy comparison chart

            for idx, (strategy_name, meta_data, json_path) in enumerate(backtest_files):
                trades_df, results, error = extract_trades(json_path)
                if not error:
                    results["Details"] = f"?strategy={strategy_name}&json_path={json_path}"
                    all_results.append(results)
                    all_trades_data[strategy_name] = trades_df
                else:
                    all_trades_data[strategy_name] = pd.DataFrame()
                pbar.progress((idx + 1) / total_count, text=f"Scanning {strategy_name}")

            pbar.empty()

            if all_results:
                tab1, tab2 = st.tabs(["Overview Table", "Comparison Chart"])

                with tab1:
                    st.subheader("Overview of Backtest Results")
                    results_df = pd.DataFrame(all_results).fillna("N/A")
                    if "Optimized" in results_df.columns:
                        results_df.drop(columns=["Optimized"], inplace=True)

                    st.data_editor(
                        results_df,
                        column_config={
                            "Details": st.column_config.LinkColumn("Details", display_text="View Details")
                        },
                        hide_index=True,
                    )

                with tab2:
                    st.subheader("Strategy Comparison")
                    fig_compare = compare_strategies(all_trades_data)
                    st.plotly_chart(fig_compare, use_container_width=True)

    else:
        # We are in details mode
        strategy_name = query_params["strategy"]
        json_path = query_params["json_path"]
        trades_df, results, error = extract_trades(json_path)

        if error:
            st.error(error)
            return

        st.title(f"Details for {strategy_name}")
        st.subheader("Backtest Summary")
        detail_df = pd.DataFrame([results]).fillna("N/A").T.rename(columns={0: "Value"})
        if "Optimized" in detail_df.index:
            detail_df.drop(index=["Optimized"], inplace=True)
        st.table(detail_df)

        st.subheader("Trade Data")
        st.dataframe(trades_df[["pair", "profit_abs", "open_date", "close_date"]])

        st.subheader("Interactive Profit Over Time")
        trades_df = trades_df.sort_values("open_date")
        trades_df["cumulative_profit"] = trades_df["profit_abs"].cumsum()
        fig = px.line(
            trades_df,
            x="open_date",
            y="cumulative_profit",
            title="Cumulative Profit Over Time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown Chart")
        fig_dd = draw_drawdown_chart(trades_df, strategy_name)
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Win/Loss Ratio")
        fig_wl = draw_winloss_ratio_chart(trades_df, strategy_name)
        st.plotly_chart(fig_wl, use_container_width=True)

        st.subheader("Cumulative Profit & Win/Draw/Loss Counts")
        df_sorted = trades_df.sort_values("open_date").copy()
        df_sorted["cumulative_wins"] = (df_sorted["profit_abs"] > 0).cumsum()
        df_sorted["cumulative_draws"] = (df_sorted["profit_abs"] == 0).cumsum()
        df_sorted["cumulative_losses"] = (df_sorted["profit_abs"] < 0).cumsum()
        df_sorted["cumulative_profit"] = df_sorted["profit_abs"].cumsum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df_sorted["open_date"], y=df_sorted["cumulative_wins"],
                name="Cumulative Wins", line=dict(color="green")
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=df_sorted["open_date"], y=df_sorted["cumulative_draws"],
                name="Cumulative Draws", line=dict(color="blue")
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=df_sorted["open_date"], y=df_sorted["cumulative_losses"],
                name="Cumulative Losses", line=dict(color="red")
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=df_sorted["open_date"], y=df_sorted["cumulative_profit"],
                name="Cumulative Profit", line=dict(color="gold")
            ),
            secondary_y=True
        )
        # Mark bull market top as a datetime
        bull_market_top = datetime(2021, 11, 10).timestamp()
        fig.add_vline(
            x=bull_market_top,
            line=dict(color="yellow", dash="dash"),
            annotation_text="2021 Bull market top",
            annotation_position="top right"
        )
        fig.update_xaxes(type="date")
        fig.update_xaxes(title_text="Time")
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

        st.subheader("Winrate & Profit Distributions")
        fig_dist = draw_distribution_charts(trades_df)
        st.plotly_chart(fig_dist, use_container_width=True)


if __name__ == "__main__":
    main()
