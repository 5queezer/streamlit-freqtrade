import plotly.graph_objects as go


def compare_strategies(all_strategies):
    """
    Renders a multi-line chart comparing cumulative profits
    for each strategy's trades, grouped weekly.
    Adds a vertical line for the 2021 bull market top as a datetime.
    """
    # bull_market_top = datetime(2021, 11, 10).timestamp()
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

    # fig.add_vline(
    #     x=bull_market_top,
    #     line=dict(color="yellow", dash="dash"),
    #     annotation_text="2021 Bull market top",
    #     annotation_position="top right"
    # )
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
