import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from charts.allcharts import (draw_drawdown_chart, draw_winloss_ratio_chart, draw_distribution_charts,
                              draw_cumulative_profit_chart)
from utils.trades import extract_trades


def show_details(query_params: dict):
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

    st.subheader("Profit Over Time")
    fig_cp = draw_cumulative_profit_chart(trades_df, strategy_name)
    st.plotly_chart(fig_cp, use_container_width=True)

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
    # bull_market_top = datetime(2021, 11, 10).timestamp()
    # fig.add_vline(
    #     x=bull_market_top,
    #     line=dict(color="yellow", dash="dash"),
    #     annotation_text="2021 Bull market top",
    #     annotation_position="top right"
    # )
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
    fig_dist = draw_distribution_charts(trades_df, strategy_name)
    st.plotly_chart(fig_dist, use_container_width=True)
