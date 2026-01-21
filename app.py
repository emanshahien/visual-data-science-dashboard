import pandas as pd
from dash import Dash, dcc, html, Input, Output, ctx
import plotly.express as px

df = pd.read_csv("countries_dashboard.csv").reset_index(drop=True)
df["row_id"] = df.index  

FEATURES = [
    "GDP ($ per capita)",
    "Literacy (%)",
    "Infant mortality (per 1000 births)",
    "Phones (per 1000)",
    "Birthrate",
    "Deathrate",
]

app = Dash(__name__)
server = app.server 

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "16px"},
    children=[
        html.H2("Global Development Dashboard (K-Means Clustering)"),

        html.Div(
            "Click a point on PCA scatter OR use box/lasso selection in the PCA scatter. Other charts update automatically.",
            style={"color": "#444", "maxWidth": "1000px", "marginBottom": "8px"},
        ),

        html.Button(
            "Reset selection",
            id="reset_btn",
            n_clicks=0,
            style={
                "marginBottom": "12px",
                "padding": "8px 14px",
                "borderRadius": "6px",
                "border": "1px solid #ccc",
                "cursor": "pointer",
                "backgroundColor": "#f5f5f5",
            },
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "10px"},
            children=[
                html.Div(
                    style={"minWidth": "260px"},
                    children=[
                        html.Label("Histogram metric"),
                        dcc.Dropdown(
                            id="metric",
                            options=[{"label": c, "value": c} for c in FEATURES],
                            value="GDP ($ per capita)",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "260px"},
                    children=[
                        html.Label("Region filter (global)"),
                        dcc.Dropdown(
                            id="region",
                            options=[{"label": r, "value": r} for r in sorted(df["Region"].dropna().unique())],
                            value=None,
                            placeholder="All regions",
                            clearable=True,
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "16px"},
            children=[
                dcc.Graph(id="pca_scatter", style={"height": "430px"}),
                dcc.Graph(id="histogram", style={"height": "430px"}),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="cluster_profile", style={"height": "420px"}),
                dcc.Graph(id="region_counts", style={"height": "420px"}),
            ],
        ),
    ],
)


def apply_selection(df_in, selectedData, clickData):
    if selectedData and "points" in selectedData and len(selectedData["points"]) > 0:
        ids = [p["customdata"][0] for p in selectedData["points"] if p.get("customdata")]
        if ids:
            return df_in[df_in["row_id"].isin(ids)]

    if clickData and "points" in clickData and len(clickData["points"]) > 0:
        pid = clickData["points"][0]["customdata"][0]
        return df_in[df_in["row_id"] == pid]
    return df_in

@app.callback(
    Output("pca_scatter", "figure"),
    Output("histogram", "figure"),
    Output("cluster_profile", "figure"),
    Output("region_counts", "figure"),
    Output("pca_scatter", "selectedData"),  
    Output("pca_scatter", "clickData"),     
    Input("pca_scatter", "selectedData"),
    Input("pca_scatter", "clickData"),
    Input("metric", "value"),
    Input("region", "value"),
    Input("reset_btn", "n_clicks"),
)
def update(selectedData, clickData, metric, region_value, reset_clicks):
    triggered = ctx.triggered_id
    base = df.copy()
    if region_value:
        base = base[base["Region"] == region_value]

    if triggered == "reset_btn":
        selectedData = None
        clickData = None

    sel = apply_selection(base, selectedData, clickData)

    fig_scatter = px.scatter(
        base,
        x="pc1",
        y="pc2",
        color="cluster_label",
        hover_data=["Country", "Region"] + FEATURES,
        custom_data=["row_id"],  
        title="PCA Scatter (click or box/lasso select to filter other charts)"
    )
    fig_scatter.update_traces(marker={"size": 9, "opacity": 0.85})
    fig_scatter.update_layout(legend_title_text="Cluster")

    fig_hist = px.histogram(
        sel,
        x=metric,
        color="cluster_label",
        nbins=30,
        title=f"Distribution of {metric} (filtered)"
    )
    fig_hist.update_layout(legend_title_text="Cluster")
    fig_hist.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>"
                      + f"{metric}: %{{x}}<br>"
                      + "Count of countries: %{y}<extra></extra>"
    )

    prof = sel.groupby("cluster_label")[FEATURES].mean().reset_index()
    prof_long = prof.melt(id_vars="cluster_label", var_name="feature", value_name="mean")
    fig_prof = px.bar(
        prof_long,
        x="feature",
        y="mean",
        color="cluster_label",
        barmode="group",
        title="Cluster profile (mean values, filtered)"
    )
    fig_prof.update_xaxes(tickangle=25)
    fig_prof.update_layout(legend_title_text="Cluster", xaxis_title="")

    reg = sel.groupby(["Region", "cluster_label"]).size().reset_index(name="count")
    fig_reg = px.bar(
        reg,
        x="Region",
        y="count",
        color="cluster_label",
        barmode="stack",
        title="Selected countries by region (stacked)"
    )
    fig_reg.update_xaxes(tickangle=25)
    fig_reg.update_layout(legend_title_text="Cluster", xaxis_title="")

    return fig_scatter, fig_hist, fig_prof, fig_reg, selectedData, clickData

if __name__ == "__main__":
    app.run(debug=True)
