import streamlit as st
import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt


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
    }

    return trades_df, results, None


def plot_balance(trades_df, strategy_name):
    trades_df['cumulative_profit'] = trades_df['profit_abs'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(trades_df['cumulative_profit'], label=f"{strategy_name} Cumulative Profit")
    plt.xlabel("Trades")
    plt.ylabel("Profit")
    plt.title(f"Balance Over Time - {strategy_name}")
    plt.legend()
    st.pyplot(plt)


def main():
    st.title("Freqtrade Multi-Backtest Analyzer")
    folder_dir = os.environ.get('BACKTEST_FOLDER', "./backtests")
    folder = st.text_input("Enter folder path containing backtest results:", folder_dir)

    if st.button("Scan Folder"):
        if not os.path.exists(folder):
            st.error("Folder not found.")
            return

        backtest_files = find_backtest_files(folder)

        if not backtest_files:
            st.warning("No valid backtest files found.")
            return

        all_results = []

        for strategy_name, meta_data, json_path in backtest_files:
            trades_df, results, error = extract_trades(json_path)

            if error:
                st.error(f"{strategy_name}: {error}")
            else:
                all_results.append(results)
                st.subheader(f"Strategy: {strategy_name}")
                st.json(results)

                st.subheader("Balance Over Time")
                plot_balance(trades_df, strategy_name)

                st.subheader("Trade Data")
                st.dataframe(trades_df[['pair', 'profit_abs', 'open_date', 'close_date']])

        if all_results:
            st.subheader("Comparison of Backtest Results")
            results_df = pd.DataFrame(all_results)
            st.dataframe(results_df)


if __name__ == "__main__":
    main()
