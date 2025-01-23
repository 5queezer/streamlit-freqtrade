import glob
import json
import math
import os
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="Freqtrade Multi-Backtest Analyzer")

# Remove scrollbars and expand tables
st.markdown(
    """
    <style>
    /* Remove vertical scrollbars on tables */
    .block-container .dataframe-container {
        max-height: none !important;
        overflow-y: visible !important;
    }
    /* Example styling for table headers (dark background, centered text, white font) */
    thead tr th {
        background-color: #1f2f3f !important;
        color: #ffffff !important;
        text-align: center !important;
    }
    /* Example styling for table body (slightly lighter background, center text, lighter font) */
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
    """Compute the max and average win/loss streak from the trades."""
    sorted_df = trades_df.sort_values("open_date")
    outcomes = (sorted_df["profit_abs"] > 0).astype(int).tolist()

    win_count, max_win_streak, win_streaks = 0, 0, []
    loss_count, max_loss_streak, loss_streaks = 0, 0, []

    for outcome in outcomes:
        if outcome == 1:
            win_count += 1
            max_win_streak = max(max_win_streak, win_count)
            if loss_count > 0:
                loss_streaks.append(loss_count)
            loss_count = 0
        else:
            loss_count += 1
            max_loss_streak = max(max_loss_streak, loss_count)
            if win_count > 0:
                win_streaks.append(win_count)
            win_count = 0

    if win_count > 0:
        win_streaks.append(win_count)
    if loss_count > 0:
        loss_streaks.append(loss_count)

    win_avg = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    loss_avg = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

    return {
        "Winstr. Max": max_win_streak,
        "Winstr. Avg": round(win_avg, 1),
        "Losestr. Max": max_loss_streak,
        "Losestr. Avg": round(loss_avg, 1),
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

    # Basic metrics
    start_date = trades_df["open_date"].min()
    end_date = trades_df["close_date"].max()
    days_diff = (end_date - start_date).days + 1 if not pd.isnull(end_date) else 0
    total_profit = trades_df["profit_abs"].sum()

    # Assume a starting balance of 10000 if not provided
    start_balance = 10000.0
    end_balance = start_balance + total_profit
    profit_pct = ((end_balance - start_balance) / start_balance) * 100

    # Additional metrics from JSON
    max_drawdown = strategy_data.get("max_drawdown", float("nan"))
    cagr = strategy_data.get("cagr", float("nan"))
    sortino = strategy_data.get("sortino", float("nan"))
    sharpe = strategy_data.get("sharpe", float("nan"))

    # Compute calmar ratio if possible: cagr / drawdown
    try:
        if (not math.isnan(float(cagr))) and (not math.isnan(float(max_drawdown))) and float(max_drawdown) != 0:
            calmar_ratio = float(cagr) / float(max_drawdown)
        else:
            calmar_ratio = float("nan")
    except:
        calmar_ratio = float("nan")

    # Profit factor = sum of gains / absolute sum of losses
    positive_sum = trades_df[trades_df["profit_abs"] > 0]["profit_abs"].sum()
    negative_sum = trades_df[trades_df["profit_abs"] < 0]["profit_abs"].sum()
    if negative_sum != 0:
        profit_factor = positive_sum / abs(negative_sum)
    else:
        profit_factor = float("nan")

    # Pairs % example: how many different pairs used vs. total pairs?
    pair_counts = trades_df["pair"].nunique()
    # Hard to define "total pairs" in a vacuum, so let's guess 50
    pairs_pct = (pair_counts / 50) * 100

    # Win rate
    win_rate = (trades_df["profit_abs"] > 0).mean() * 100

    # Streaks
    streaks = compute_streaks(trades_df)

    # Weighted total score (arbitrary example)
    # Feel free to adjust weighting as desired
    score = (
            (profit_pct * 0.2)
            + (win_rate * 0.2)
            + (float(sortino) if not math.isnan(float(sortino)) else 0) * 0.2
            + (float(sharpe) if not math.isnan(float(sharpe)) else 0) * 0.2
            - (float(max_drawdown) if not math.isnan(float(max_drawdown)) else 0) * 0.2
    )

    results = {
        "TF": strategy_data.get("timeframe", strategy_name),
        "End Balance": round(end_balance, 2),
        "Profit %": round(profit_pct, 2),
        "Win %": round(win_rate, 2),
        "Trades": len(trades_df),
        "Winstr. Max": streaks["Winstr. Max"],
        "Winstr. Avg": streaks["Winstr. Avg"],
        "Losestr. Max": streaks["Losestr. Max"],
        "Losestr. Avg": streaks["Losestr. Avg"],
        "CAGR": round(float(cagr), 2) if not math.isnan(float(cagr)) else "N/A",
        "Max Drawdown": round(float(max_drawdown), 2) if not math.isnan(float(max_drawdown)) else "N/A",
        "Calmar Ratio": round(calmar_ratio, 2) if not math.isnan(calmar_ratio) else "N/A",
        "Sortino": round(float(sortino), 2) if not math.isnan(float(sortino)) else "N/A",
        "Sharpe": round(float(sharpe), 2) if not math.isnan(float(sharpe)) else "N/A",
        "Profit Factor": round(profit_factor, 2) if not math.isnan(profit_factor) else "N/A",
        "Pairs %": round(pairs_pct, 2),
        "Total Score": round(score, 1),
        "Optimized": 0,
        "Start Date": str(start_date.date()) if not pd.isnull(start_date) else "N/A",
        "End Date": str(end_date.date()) if not pd.isnull(end_date) else "N/A",
        "Days": days_diff,
    }
    return trades_df, results, None


def main():
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

        for idx, (strategy_name, meta_data, json_path) in enumerate(backtest_files):
            trades_df, results, error = extract_trades(json_path)
            if not error:
                results["Details"] = f"?strategy={strategy_name}&json_path={json_path}"
                all_results.append(results)
            pbar.progress((idx + 1) / total_count, text=f"Scanning {strategy_name}")

        pbar.empty()
        if all_results:
            st.subheader("Overview of Backtest Results")
            results_df = pd.DataFrame(all_results).fillna("N/A")
            # Show as a data editor
            st.data_editor(
                results_df,
                column_config={"Details": st.column_config.LinkColumn("Details", display_text="View Details")},
                hide_index=True,
            )

    # Handle details
    query_params = st.query_params
    if "strategy" in query_params and "json_path" in query_params:
        strategy_name = query_params["strategy"]
        json_path = query_params["json_path"]
        trades_df, results, error = extract_trades(json_path)

        if error:
            st.error(error)
        else:
            st.title(f"Details for {strategy_name}")
            st.subheader("Backtest Summary")

            detail_df = pd.DataFrame([results]).fillna("N/A").T.rename(columns={0: "Value"})
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


if __name__ == "__main__":
    main()
