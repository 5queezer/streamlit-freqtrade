import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def find_backtest_files(folder: str):
    meta_files = glob.glob(os.path.join(folder, "*.meta.json"))
    json_files = {os.path.basename(f): f for f in glob.glob(os.path.join(folder, r"*[!meta\.].json"))}

    backtest_data = []

    for meta_file in meta_files:
        meta_data = load_json(meta_file)
        strategy_name = list(meta_data.keys())[0] if meta_data else None

        if not strategy_name:
            continue

        json_file = os.path.basename(meta_file).replace(".meta.json", ".json")
        json_path = json_files.get(json_file)

        if json_path and os.path.exists(json_path):
            backtest_data.append((strategy_name, meta_data[strategy_name], json_path))

    return backtest_data


def extract_trades(json_path):
    data = load_json(json_path)
    strategy_name = list(data.get("strategy", {}).keys())[0] if "strategy" in data else None
    if not strategy_name:
        return None, None, "Invalid JSON: No strategy key found."

    strategy_data = data["strategy"].get(strategy_name, {})
    trades = strategy_data.get("trades", [])

    if not trades:
        return None, None, "No trades found in the backtest data."

    trades_df = pd.DataFrame(trades)
    trades_df['profit_abs'] = trades_df['profit_abs'].astype(float)

    results = {
        "Strategy": strategy_name,
        "Total Trades": len(trades_df),
        "Win Rate (%)": (trades_df['profit_abs'] > 0).mean() * 100,
        "Total Profit": trades_df['profit_abs'].sum(),
        "Avg Profit per Trade": trades_df['profit_abs'].mean(),
        "Max Drawdown": strategy_data.get('max_drawdown', "N/A"),
        "Best Pair": trades_df.groupby('pair')['profit_abs'].sum().idxmax(),
        "Worst Pair": trades_df.groupby('pair')['profit_abs'].sum().idxmin(),
        "CAGR": strategy_data.get('cagr', "N/A"),
        "Sortino": strategy_data.get('sortino', "N/A"),
        "Sharpe": strategy_data.get('sharpe', "N/A"),
    }

    return trades_df, results, None


def main():
    st.title("Freqtrade Multi-Backtest Analyzer")
    folder_dir = os.environ.get('BACKTEST_FOLDER', "./backtests")
    folder = st.text_input("Enter folder path containing backtest results:", folder_dir)

    if st.button("Scan Folder"):
        if not os.path.exists(folder):
            st.error("Folder not found.")
            return

        progress_text = "Operation in progress. Please wait."
        pbar = st.progress(0, text=progress_text)

        backtest_files = find_backtest_files(folder)

        if not backtest_files:
            st.warning("No valid backtest files found.")
            return

        all_results = []
        total_count = len(backtest_files)

        for idx, (strategy_name, meta_data, json_path) in enumerate(backtest_files):
            trades_df, results, error = extract_trades(json_path)

            if error:
                pass
            else:
                results["Details"] = f"details?strategy={strategy_name}&json_path={json_path}"
                all_results.append(results)

            progress_text = f"Scanning {strategy_name}"
            percent_complete = (idx + 1) / total_count
            pbar.progress(percent_complete, text=progress_text)

        pbar.empty()

        if all_results:
            st.subheader("Overview of Backtest Results")
            results_df = pd.DataFrame(all_results)
            st.data_editor(results_df, column_config={
                "Details": st.column_config.LinkColumn("Details", display_text="View Details")
            }, hide_index=True)

    query_params = st.query_params
    if "strategy" in query_params and "json_path" in query_params:
        strategy_name = query_params["strategy"]
        json_path = query_params["json_path"]
        trades_df, results, error = extract_trades(json_path)

        if error:
            st.error(error)
        else:
            st.title(f"Details for {strategy_name}")
            st.json(results)
            st.subheader("Trade Data")
            st.dataframe(trades_df[['pair', 'profit_abs', 'open_date', 'close_date']])

            # Plot profit over time
            st.subheader("Cumulative Profit Over Time")
            trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
            trades_df = trades_df.sort_values("open_date")
            trades_df["cumulative_profit"] = trades_df["profit_abs"].cumsum()

            fig, ax = plt.subplots()
            ax.plot(trades_df["open_date"], trades_df["cumulative_profit"], marker='o', linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Profit")
            ax.set_title("Cumulative Profit Over Time")
            ax.grid()

            st.pyplot(fig)


if __name__ == "__main__":
    main()
