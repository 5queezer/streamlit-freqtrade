import glob
import os
import pickle
import hashlib

import pandas as pd
import streamlit as st

from tabs.compare import compare_strategies
from tabs.details import show_details
from utils.json import load_json
from utils.trades import extract_trades

st.set_page_config(layout="wide", page_title="Freqtrade Multi-Backtest Analyzer")

CACHE_FILE = "backtest_cache.pkl"
TRADES_CACHE_FILE = "trades_data_cache.pkl"
RESULTS_CACHE_FILE = "results_cache.pkl"


def compute_folder_hash(folder: str) -> str:
    """Computes a hash of all file modification times in the folder for cache invalidation."""
    file_metadata = [(filepath, os.path.getmtime(filepath)) for filepath in glob.glob(os.path.join(folder, "*")) if os.path.exists(filepath)]
    return hashlib.md5(str(file_metadata).encode()).hexdigest()


def load_cache(cache_file: str, folder: str):
    """Loads cached data if cache exists and is still valid."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and "folder_hash" in cached_data and "data" in cached_data:
                if cached_data["folder_hash"] == compute_folder_hash(folder):
                    return cached_data["data"]
        except (EOFError, pickle.UnpicklingError, KeyError):
            st.warning(f"Cache file `{cache_file}` corrupted. Recomputing...")
    return None


def save_cache(cache_file: str, folder: str, data):
    """Saves data to cache."""
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"folder_hash": compute_folder_hash(folder), "data": data}, f)
    except Exception as e:
        st.warning(f"Failed to save cache `{cache_file}`: {e}")


def find_backtest_files(folder: str):
    """Finds and maps backtest JSON and metadata files in a given folder."""
    cached_data = load_cache(CACHE_FILE, folder)
    if cached_data is not None:
        return cached_data  # Return cached backtest files directly

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
        json_path = file_map.get(guessed_json) or file_map.get(guessed_json.replace(".json", ".zip"))

        if json_path and os.path.exists(json_path):
            backtest_data.append((strategy_name, meta_data[strategy_name], json_path))

    save_cache(CACHE_FILE, folder, backtest_data)
    return backtest_data


def load_trades_data(folder: str, backtest_files: list):
    """Loads cached trade data if available, otherwise extracts it from backtest files."""
    cached_trades_data = load_cache(TRADES_CACHE_FILE, folder)
    if cached_trades_data is not None:
        return cached_trades_data, True  # Return cached data and flag as cached

    all_trades_data = {}
    for strategy_name, meta_data, json_path in backtest_files:
        trades_df, _, error = extract_trades(json_path)
        all_trades_data[strategy_name] = trades_df if not error else pd.DataFrame()

    save_cache(TRADES_CACHE_FILE, folder, all_trades_data)
    return all_trades_data, False  # Return new data and flag as not cached


def load_results_cache(folder: str):
    """Loads cached results data for the overview table."""
    cached_results = load_cache(RESULTS_CACHE_FILE, folder)
    return cached_results if cached_results else []


def save_results_cache(folder: str, results: list):
    """Saves results data for the overview table."""
    save_cache(RESULTS_CACHE_FILE, folder, results)


def main():
    query_params = st.query_params
    in_details = "strategy" in query_params and "json_path" in query_params

    if not in_details:
        st.title("Freqtrade Multi-Backtest Analyzer")
        folder_dir = os.environ.get("BACKTEST_FOLDER", "./backtests")
        folder = st.text_input("Enter folder path containing backtest results:", folder_dir)

        if st.button("Scan Folder"):
            if not os.path.exists(folder):
                st.error("Folder not found.")
                return

            all_results = load_results_cache(folder)  # Load cached overview table

            with st.status("Checking cache...", expanded=True) as status:
                backtest_files = find_backtest_files(folder)

                if not backtest_files:
                    status.update(label="No valid backtest files found.", state="error")
                    return

                all_trades_data, is_cached = load_trades_data(folder, backtest_files)

                if is_cached and all_results:
                    status.update(label="Cache loaded successfully. Skipping extraction.", state="complete")
                else:
                    status.update(label="Cache not found. Extracting trades...", state="running")
                    all_results = []  # Reset results since new extraction is needed

                    total_count = len(backtest_files)
                    pbar = st.progress(0, text="Scanning...")

                    for idx, (strategy_name, meta_data, json_path) in enumerate(backtest_files):
                        trades_df = all_trades_data.get(strategy_name, pd.DataFrame())
                        if not trades_df.empty:
                            _, results, error = extract_trades(json_path)
                            if not error:
                                results["Details"] = f"?strategy={strategy_name}&json_path={json_path}"
                                all_results.append(results)

                        pbar.progress((idx + 1) / total_count, text=f"Scanning {strategy_name}")

                    pbar.empty()
                    status.update(label="Trade extraction complete.", state="complete")

                    save_results_cache(folder, all_results)  # Save extracted results to cache

            # Ensure UI elements are populated
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
        show_details(query_params)


if __name__ == "__main__":
    main()
