import pandas as pd


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
