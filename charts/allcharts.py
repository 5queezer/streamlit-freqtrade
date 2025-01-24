import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the dataframe by 'open_date' and computes cumulative profit.
    """
    df_sorted = df.sort_values("open_date").copy()
    df_sorted["cumulative_profit"] = df_sorted["profit_abs"].cumsum()
    return df_sorted


def create_figure(title: str, xaxis_title: str, yaxis_title: str) -> go.Figure:
    """
    Creates a styled Plotly figure with a dark theme.
    """
    fig = go.Figure()
    fig.update_xaxes(type="date")
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="x unified",
    )
    return fig


def draw_cumulative_profit_chart(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """
    Plots a cumulative profit chart over time.
    """
    df_sorted = preprocess_dataframe(df)

    fig = create_figure(
        title=f"Strategy {strategy_name} Cumulative Profit Over Time",
        xaxis_title="Time",
        yaxis_title="Cumulative Profit",
    )

    fig.add_trace(
        go.Line(
            x=df_sorted["open_date"],
            y=df_sorted["cumulative_profit"],
            name="Cumulative Profit",
            mode="lines+markers",
            line=dict(color="yellow"),
        )
    )

    return fig


def draw_drawdown_chart(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """
    Plots a drawdown chart over time using the running peak logic.
    """
    df_sorted = preprocess_dataframe(df)
    df_sorted["peak"] = df_sorted["cumulative_profit"].cummax()
    df_sorted["drawdown"] = ((df_sorted["peak"] - df_sorted["cumulative_profit"]) / df_sorted["peak"]) * 100
    df_sorted["drawdown"] = df_sorted["drawdown"].fillna(0)

    avg_dd = df_sorted["drawdown"].mean()

    fig = create_figure(
        title=f"Strategy {strategy_name} Drawdown Analysis",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["open_date"],
            y=df_sorted["drawdown"],
            name="Drawdown (%)",
            line=dict(color="red"),
        )
    )

    fig.add_hline(
        y=avg_dd,
        line=dict(color="green"),
        annotation_text=f"Average Drawdown: {round(avg_dd, 2)}%",
        annotation_position="top left",
    )

    return fig


def draw_winloss_ratio_chart(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """
    Plots the Win/Loss ratio by week.
    """
    df_sorted = preprocess_dataframe(df)
    df_sorted.set_index("open_date", inplace=True)

    weekly = pd.DataFrame()
    weekly["wins"] = df_sorted["profit_abs"].resample("W").apply(lambda x: (x > 0).sum())
    weekly["losses"] = df_sorted["profit_abs"].resample("W").apply(lambda x: (x < 0).sum())

    weekly["net_wins"] = weekly["wins"] - weekly["losses"]
    weekly["ratio"] = weekly.apply(lambda row: row["wins"] / row["losses"] if row["losses"] > 0 else float("inf"),
                                   axis=1)

    fig = create_figure(
        title=f"Strategy {strategy_name}: Win/Loss Ratio by Week",
        xaxis_title="Time",
        yaxis_title="Win/Loss Ratio",
    )

    fig.add_trace(go.Bar(x=weekly.index, y=weekly["ratio"], name="Win/Loss Ratio", marker_color="green"))

    return fig


def draw_distribution_charts(df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """
    Generates two boxplots: winrate and profit distribution by trading pair.
    """
    by_pair = df.groupby("pair")
    winrates = by_pair.apply(lambda x: (x["profit_abs"] > 0).mean(), include_groups=False)
    profits = by_pair["profit_abs"].sum()

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Winrate Distribution", "Profit Distribution"])

    fig.add_trace(go.Box(y=winrates, name="Winrate", boxpoints="all", jitter=0.3, marker_color="#95DAC1"), row=1, col=1)
    fig.add_trace(go.Box(y=profits, name="Profit", boxpoints="all", jitter=0.3, marker_color="#95A7DA"), row=1, col=2)

    fig.update_layout(
        title=f"Strategy {strategy_name} Distribution Analysis",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#EEEEEE"),
        hovermode="closest",
        height=600,
    )

    return fig
