import math

import pandas as pd

from utils.json import load_json
from utils.streaks import compute_streaks


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
    days_diff = (end_date - start_date).days + 1 if pd.notnull(end_date) else float("nan")

    start_balance = 10000.0
    total_profit = trades_df["profit_abs"].sum()
    end_balance = start_balance + total_profit
    profit_pct = ((end_balance - start_balance) / start_balance) * 100 if start_balance else float("nan")

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
        "Start Balance": round(start_balance, 2),
        "End Balance": round(end_balance, 2),
        "Start Date": str(start_date.date()) if pd.notnull(start_date) else "N/A",
        "End Date": str(end_date.date()) if pd.notnull(end_date) else "N/A",
        "Days": days_diff,
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
    }
    return trades_df, results, None
