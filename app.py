from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PASSWORD = "healthanalytics2026"
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "Life Expectancy Data.csv"

TARGET = "Life Expectancy"
CATEGORICAL_COLUMNS = ["Country", "Status"]

CANONICAL_COLUMNS = {
    "country": "Country",
    "year": "Year",
    "status": "Status",
    "lifeexpectancy": "Life Expectancy",
    "adultmortality": "Adult Mortality",
    "infantdeaths": "Infant Deaths",
    "alcohol": "Alcohol",
    "percentageexpenditure": "Percentage Health Expenditure",
    "hepatitisb": "Hepatitis B",
    "measles": "Measles",
    "bmi": "BMI",
    "underfivedeaths": "Under-Five Deaths",
    "polio": "Polio",
    "totalexpenditure": "Total Expenditure",
    "diphtheria": "Diphtheria",
    "hivaids": "HIV/AIDS",
    "gdp": "GDP",
    "population": "Population",
    "thinness119years": "Thinness Age 1-19",
    "thinness19years": "Thinness Age 1-19",
    "thinness59years": "Thinness Age 5-9",
    "incomecompositionofresources": "Income Composition",
    "incomecomposition": "Income Composition",
    "schooling": "Schooling",
}

VARIABLE_DEFINITIONS = {
    "Life Expectancy": "Average number of years a newborn is expected to live under current mortality patterns.",
    "Adult Mortality": "Adult mortality rate, commonly interpreted as deaths among adults per 1,000 population.",
    "Infant Deaths": "Number of infant deaths reported for the country-year record.",
    "Under-Five Deaths": "Number of deaths among children under five reported for the country-year record.",
    "HIV/AIDS": "Deaths or burden associated with HIV/AIDS in the dataset source.",
    "BMI": "Average body mass index indicator.",
    "Hepatitis B": "Hepatitis B immunization coverage.",
    "Polio": "Polio immunization coverage.",
    "Diphtheria": "Diphtheria immunization coverage.",
    "Percentage Health Expenditure": "Health expenditure as a percentage measure in the source file.",
    "Total Expenditure": "Total health expenditure indicator.",
    "GDP": "Gross domestic product indicator.",
    "Population": "Population count.",
    "Income Composition": "Index measuring income composition of resources.",
    "Schooling": "Average years of schooling.",
    "Status": "Development status category, usually Developed or Developing.",
}


st.set_page_config(
    page_title="Life Expectancy Analytics Dashboard",
    page_icon="health",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7fafc;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #334e68;
        font-size: 0.88rem;
    }
    .section-note {
        border-left: 4px solid #2a9d8f;
        background: #f3fbf9;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #243b53;
        margin: 0.5rem 0 1rem 0;
    }
    .small-muted {color: #5c6f82; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def password_gate() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("Life Expectancy Analytics Dashboard")
    st.caption("Population health, healthcare system, and socioeconomic drivers")
    password = st.text_input("Enter dashboard password", type="password")
    if st.button("Open dashboard", type="primary"):
        if password == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.info("Default project password: healthanalytics2026")
    return False


def normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column_name).strip().lower())


def pretty_unknown_column(column_name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(column_name).strip())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Unknown Column"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        key = normalize_column_name(column)
        rename_map[column] = CANONICAL_COLUMNS.get(key, pretty_unknown_column(column))

    clean_df = df.rename(columns=rename_map)
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()].copy()
    return clean_df


def read_uploaded_or_default(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file), f"Uploaded file: {uploaded_file.name}"
        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), f"Uploaded file: {uploaded_file.name}"
        st.error("Please upload a CSV or Excel file.")
        return None, "Unsupported file"

    if DEFAULT_DATA_PATH.exists():
        return pd.read_csv(DEFAULT_DATA_PATH), f"Bundled sample file: {DEFAULT_DATA_PATH.name}"

    return None, "No file loaded"


@st.cache_data(show_spinner=False)
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw_df)

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().fillna("Unknown")

    numeric_columns = [c for c in df.columns if c not in CATEGORICAL_COLUMNS]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if TARGET not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=[TARGET]).copy()

    for column in [c for c in df.columns if c not in CATEGORICAL_COLUMNS and c != TARGET]:
        median_value = df[column].median()
        if pd.isna(median_value):
            median_value = 0
        df[column] = df[column].fillna(median_value)

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown").replace("", "Unknown")

    if "Year" in df.columns:
        df["Year"] = df["Year"].round().astype("Int64")

    return df


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def format_number(value, decimals: int = 1, prefix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.{decimals}f}B"
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:.{decimals}f}K"
    return f"{prefix}{value:.{decimals}f}"


def apply_filters(df: pd.DataFrame, year_range, countries: list[str], statuses: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    if "Year" in filtered.columns and year_range:
        filtered = filtered[
            (filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])
        ]
    if countries and "Country" in filtered.columns:
        filtered = filtered[filtered["Country"].isin(countries)]
    if statuses and "Status" in filtered.columns:
        filtered = filtered[filtered["Status"].isin(statuses)]
    return filtered


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], title: str):
    cols = available_columns(df, columns)
    if len(cols) < 2:
        st.warning("Not enough available numeric columns to build this correlation view.")
        return

    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_traces(
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>",
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def scatter(df: pd.DataFrame, x: str, y: str = TARGET, title: str | None = None):
    if x not in df.columns or y not in df.columns:
        st.warning(f"{x} is not available in the uploaded dataset.")
        return

    color = "Status" if "Status" in df.columns else None
    hover = ["Country", "Year"] if all(c in df.columns for c in ["Country", "Year"]) else None
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=hover,
        trendline="ols" if len(df) >= 20 else None,
        title=title or f"{x} vs {y}",
        opacity=0.72,
    )
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def line_by_year(df: pd.DataFrame, metric: str, title: str, color: str | None = None):
    if "Year" not in df.columns or metric not in df.columns:
        st.warning(f"{metric} trend cannot be displayed because a required column is missing.")
        return
    group_cols = ["Year"] + ([color] if color and color in df.columns else [])
    trend = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()
    fig = px.line(trend, x="Year", y=metric, color=color, markers=True, title=title)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def insight_box(text: str):
    st.markdown(f"<div class='section-note'>{text}</div>", unsafe_allow_html=True)


def strongest_relationships(df: pd.DataFrame, candidates: list[str], positive: bool = False) -> pd.Series:
    cols = available_columns(df, [TARGET] + candidates)
    if len(cols) < 2:
        return pd.Series(dtype=float)
    correlations = df[cols].corr(numeric_only=True)[TARGET].drop(TARGET, errors="ignore")
    correlations = correlations.dropna()
    if positive:
        return correlations.sort_values(ascending=False)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def executive_overview(df: pd.DataFrame):
    st.header("Executive Overview")
    if df.empty:
        st.warning("No records match the selected filters.")
        return

    country_avg = (
        df.groupby("Country", dropna=False)[TARGET].mean().sort_values(ascending=False)
        if "Country" in df.columns
        else pd.Series(dtype=float)
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Average Life Expectancy", format_number(df[TARGET].mean()))
    if not country_avg.empty:
        metric_cols[1].metric("Highest Country", f"{country_avg.index[0]}", format_number(country_avg.iloc[0]))
        metric_cols[2].metric("Lowest Country", f"{country_avg.index[-1]}", format_number(country_avg.iloc[-1]))
    else:
        metric_cols[1].metric("Highest Country", "N/A")
        metric_cols[2].metric("Lowest Country", "N/A")
    metric_cols[3].metric("Countries", int(df["Country"].nunique()) if "Country" in df.columns else "N/A")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Average Adult Mortality", format_number(df.get("Adult Mortality", pd.Series(dtype=float)).mean()))
    metric_cols[1].metric("Average Schooling", format_number(df.get("Schooling", pd.Series(dtype=float)).mean()))
    metric_cols[2].metric("Average GDP", format_number(df.get("GDP", pd.Series(dtype=float)).mean(), prefix="$"))
    if "Year" in df.columns:
        metric_cols[3].metric("Selected Year Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
    else:
        metric_cols[3].metric("Selected Year Range", "N/A")

    st.divider()
    left, right = st.columns(2)
    with left:
        if not country_avg.empty:
            top = country_avg.head(10).sort_values()
            fig = px.bar(
                top,
                x=top.values,
                y=top.index,
                orientation="h",
                labels={"x": "Average Life Expectancy", "y": "Country"},
                title="Top 10 Countries by Average Life Expectancy",
                color=top.values,
                color_continuous_scale="Teal",
            )
            fig.update_layout(height=430, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not country_avg.empty:
            bottom = country_avg.tail(10).sort_values(ascending=False)
            fig = px.bar(
                bottom,
                x=bottom.values,
                y=bottom.index,
                orientation="h",
                labels={"x": "Average Life Expectancy", "y": "Country"},
                title="Bottom 10 Countries by Average Life Expectancy",
                color=bottom.values,
                color_continuous_scale="OrRd",
            )
            fig.update_layout(height=430, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "Status" in df.columns:
            fig = px.box(df, x="Status", y=TARGET, color="Status", title="Life Expectancy Distribution by Status")
            fig.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        scatter(df, "Adult Mortality", title="Adult Mortality vs Life Expectancy")

    insight_box(
        "In the selected period, countries with higher adult mortality generally show lower life expectancy. "
        "Developed countries tend to cluster at higher life expectancy values, while socioeconomic indicators "
        "such as schooling and income composition help explain important differences between countries."
    )


def trends_over_time(df: pd.DataFrame):
    st.header("Trends Over Time")
    if df.empty:
        st.warning("No records match the selected filters.")
        return

    line_by_year(df, TARGET, "Global Average Life Expectancy by Year")
    if "Country" in df.columns and "Year" in df.columns:
        country_trend = df.groupby(["Year", "Country"], dropna=False)[TARGET].mean().reset_index()
        fig = px.line(
            country_trend,
            x="Year",
            y=TARGET,
            color="Country",
            title="Life Expectancy by Selected Country Over Time",
            markers=True,
        )
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        line_by_year(df, "Adult Mortality", "Adult Mortality by Year")
    with right:
        line_by_year(df, "Schooling", "Schooling by Year")

    if "Status" in df.columns:
        line_by_year(df, TARGET, "Average Life Expectancy by Development Status", color="Status")

    insight_box(
        "Time trends help identify whether population health is improving consistently or whether specific "
        "countries and status groups are falling behind. Use the country filter to compare national trajectories."
    )


def mortality_page(df: pd.DataFrame):
    st.header("Mortality & Disease Burden")
    if df.empty:
        st.warning("No records match the selected filters.")
        return

    left, right = st.columns(2)
    with left:
        scatter(df, "Adult Mortality")
        scatter(df, "BMI")
    with right:
        scatter(df, "HIV/AIDS")
        scatter(df, "Infant Deaths")

    disease_cols = [
        TARGET,
        "Adult Mortality",
        "Infant Deaths",
        "Under-Five Deaths",
        "HIV/AIDS",
        "Measles",
        "BMI",
        "Thinness Age 1-19",
        "Thinness Age 5-9",
    ]
    plot_correlation_heatmap(df, disease_cols, "Correlation: Life Expectancy and Disease Burden Indicators")

    if "Status" in df.columns:
        burden_cols = available_columns(
            df,
            ["Adult Mortality", "Infant Deaths", "Under-Five Deaths", "HIV/AIDS", "Measles"],
        )
        if burden_cols:
            burden = df.groupby("Status")[burden_cols].mean().reset_index()
            long_burden = burden.melt(id_vars="Status", var_name="Indicator", value_name="Average Value")
            fig = px.bar(
                long_burden,
                x="Indicator",
                y="Average Value",
                color="Status",
                barmode="group",
                title="Average Mortality and Disease Burden Indicators by Status",
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)

    relationships = strongest_relationships(
        df,
        ["Adult Mortality", "Infant Deaths", "Under-Five Deaths", "HIV/AIDS", "Measles", "BMI", "Thinness Age 1-19", "Thinness Age 5-9"],
    )
    if not relationships.empty:
        top = relationships.head(3)
        insight_box(
            "The strongest disease and mortality relationships in the current selection are: "
            + ", ".join([f"{idx} ({val:.2f})" for idx, val in top.items()])
            + ". Negative values indicate that higher burden is associated with lower life expectancy."
        )


def healthcare_page(df: pd.DataFrame):
    st.header("Healthcare System Indicators")
    if df.empty:
        st.warning("No records match the selected filters.")
        return

    left, right = st.columns(2)
    with left:
        scatter(df, "Polio")
        scatter(df, "Hepatitis B")
    with right:
        scatter(df, "Diphtheria")
        scatter(df, "Total Expenditure")

    immunization_cols = available_columns(df, ["Hepatitis B", "Polio", "Diphtheria"])
    if immunization_cols and "Year" in df.columns:
        trend = df.groupby("Year")[immunization_cols].mean().reset_index()
        long_trend = trend.melt(id_vars="Year", var_name="Indicator", value_name="Average Coverage")
        fig = px.line(
            long_trend,
            x="Year",
            y="Average Coverage",
            color="Indicator",
            markers=True,
            title="Immunization Trends Over Time",
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    if immunization_cols and "Status" in df.columns:
        coverage = df.groupby("Status")[immunization_cols].mean().reset_index()
        long_coverage = coverage.melt(id_vars="Status", var_name="Indicator", value_name="Average Coverage")
        fig = px.bar(
            long_coverage,
            x="Indicator",
            y="Average Coverage",
            color="Status",
            barmode="group",
            title="Average Immunization Coverage by Status",
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "Immunization indicators generally show a positive association with life expectancy, especially where "
        "coverage is consistently high. Expenditure indicators should be interpreted carefully because spending "
        "levels may reflect both system investment and underlying disease burden."
    )


def socioeconomic_page(df: pd.DataFrame):
    st.header("Socioeconomic Drivers")
    if df.empty:
        st.warning("No records match the selected filters.")
        return

    left, right = st.columns(2)
    with left:
        scatter(df, "Schooling")
        scatter(df, "GDP")
    with right:
        scatter(df, "Income Composition")
        if "Status" in df.columns:
            fig = px.box(df, x="Status", y=TARGET, color="Status", title="Life Expectancy by Status")
            fig.update_layout(height=440, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    socio_cols = [TARGET, "GDP", "Population", "Income Composition", "Schooling", "Adult Mortality"]
    plot_correlation_heatmap(df, socio_cols, "Correlation: Life Expectancy and Socioeconomic Indicators")

    positive = strongest_relationships(df, ["GDP", "Population", "Income Composition", "Schooling"], positive=True)
    if not positive.empty:
        insight_box(
            "In this selection, the strongest positive socioeconomic relationships are: "
            + ", ".join([f"{idx} ({val:.2f})" for idx, val in positive.head(3).items()])
            + ". Schooling and income composition are usually strong positive predictors of life expectancy."
        )


def global_map_page(df: pd.DataFrame, map_metric: str):
    st.header("Global Map")
    if df.empty:
        st.warning("No records match the selected filters.")
        return
    if "Country" not in df.columns or map_metric not in df.columns:
        st.warning("The selected map metric or country column is not available.")
        return

    map_data = df.groupby("Country", dropna=False)[map_metric].mean().reset_index()
    fig = px.choropleth(
        map_data,
        locations="Country",
        locationmode="country names",
        color=map_metric,
        hover_name="Country",
        color_continuous_scale="Viridis",
        title=f"Average {map_metric} by Country",
    )
    fig.update_layout(height=590, margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Some country names may not be recognized by the map engine, but the dashboard will continue to run.")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Top 10 Countries by {map_metric}")
        st.dataframe(map_data.sort_values(map_metric, ascending=False).head(10), width="stretch", hide_index=True)
    with right:
        st.subheader(f"Bottom 10 Countries by {map_metric}")
        st.dataframe(map_data.sort_values(map_metric, ascending=True).head(10), width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame):
    if TARGET not in df.columns or len(df) < 50:
        return None

    feature_candidates = [
        "Year",
        "Adult Mortality",
        "Infant Deaths",
        "Alcohol",
        "Percentage Health Expenditure",
        "Hepatitis B",
        "Measles",
        "BMI",
        "Under-Five Deaths",
        "Polio",
        "Total Expenditure",
        "Diphtheria",
        "HIV/AIDS",
        "GDP",
        "Population",
        "Thinness Age 1-19",
        "Thinness Age 5-9",
        "Income Composition",
        "Schooling",
    ]
    numeric_features = available_columns(df, feature_candidates)
    categorical_features = available_columns(df, ["Status"])
    features = numeric_features + categorical_features
    if not numeric_features:
        return None

    model_df = df[features + [TARGET]].dropna(subset=[TARGET]).copy()
    X = model_df[features]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical_features),
        ],
        remainder="drop",
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2, n_jobs=-1)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    predictions = rf_pipeline.predict(X_test)

    linear_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical_features),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    linear_pipeline.fit(X_train, y_train)
    linear_predictions = linear_pipeline.predict(X_test)

    feature_names = list(numeric_features)
    if categorical_features:
        cat_pipeline = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        encoded_names = cat_pipeline.named_steps["encoder"].get_feature_names_out(categorical_features)
        feature_names.extend(encoded_names.tolist())

    importances = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf_pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    results["Residual"] = results["Actual"] - results["Predicted"]

    return {
        "pipeline": rf_pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "features": features,
        "results": results,
        "importances": importances,
        "metrics": {
            "R2": r2_score(y_test, predictions),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "MAE": mean_absolute_error(y_test, predictions),
            "Linear R2": r2_score(y_test, linear_predictions),
        },
        "medians": model_df[numeric_features].median(numeric_only=True).to_dict(),
        "status_values": sorted(model_df["Status"].dropna().unique().tolist()) if "Status" in model_df.columns else [],
    }


def predictive_model_page(df: pd.DataFrame):
    st.header("Predictive Model")
    model_info = train_model(df)
    if model_info is None:
        st.warning("The dataset does not contain enough usable records or numeric predictors to train the model.")
        return

    metrics = model_info["metrics"]
    cols = st.columns(4)
    cols[0].metric("Random Forest R-squared", f"{metrics['R2']:.3f}")
    cols[1].metric("RMSE", f"{metrics['RMSE']:.2f} years")
    cols[2].metric("MAE", f"{metrics['MAE']:.2f} years")
    cols[3].metric("Linear Regression R-squared", f"{metrics['Linear R2']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            model_info["results"],
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Life Expectancy",
            opacity=0.72,
        )
        min_val = min(model_info["results"]["Actual"].min(), model_info["results"]["Predicted"].min())
        max_val = max(model_info["results"]["Actual"].max(), model_info["results"]["Predicted"].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Perfect prediction"))
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            model_info["results"],
            x="Predicted",
            y="Residual",
            title="Residual Plot",
            opacity=0.72,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#34495e")
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

    top_importances = model_info["importances"].head(10).sort_values("Importance")
    fig = px.bar(
        top_importances,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Random Forest Predictors",
        color="Importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=470, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interactive Life Expectancy Prediction")
    st.caption("Values not shown in the form are filled with the dataset median before prediction.")
    medians = model_info["medians"]
    editable_features = available_columns(
        df,
        [
            "Year",
            "Adult Mortality",
            "BMI",
            "Schooling",
            "GDP",
            "Income Composition",
            "HIV/AIDS",
            "Polio",
            "Diphtheria",
            "Hepatitis B",
            "Total Expenditure",
        ],
    )
    input_values = medians.copy()
    form_cols = st.columns(3)
    for idx, feature in enumerate(editable_features):
        series = df[feature].dropna()
        default = float(medians.get(feature, series.median() if not series.empty else 0))
        min_value = float(series.min()) if not series.empty else 0.0
        max_value = float(series.max()) if not series.empty else max(default, 1.0)
        if min_value == max_value:
            max_value = min_value + 1.0
        with form_cols[idx % 3]:
            if feature == "Year":
                input_values[feature] = st.number_input(
                    feature,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default),
                    step=1,
                )
            else:
                input_values[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=min(max(default, min_value), max_value),
                    step=(max_value - min_value) / 100 if max_value > min_value else 1.0,
                )

    prediction_row = {}
    for feature in model_info["numeric_features"]:
        prediction_row[feature] = input_values.get(feature, medians.get(feature, 0))
    if "Status" in model_info["categorical_features"]:
        status_values = model_info["status_values"] or ["Unknown"]
        prediction_row["Status"] = st.selectbox("Development Status", status_values)

    predicted_value = model_info["pipeline"].predict(pd.DataFrame([prediction_row]))[0]
    st.success(f"Predicted Life Expectancy: {predicted_value:.1f} years")

    insight_box(
        "The prediction model suggests that adult mortality, schooling, income composition, HIV/AIDS burden, "
        "and immunization-related indicators are commonly important predictors. This is an analytical model for "
        "population-level planning, not a causal or patient-level clinical tool."
    )


def methodology_page(raw_df: pd.DataFrame, clean_df: pd.DataFrame, source_label: str):
    st.header("Data Dictionary & Methodology")
    st.subheader("Dataset")
    st.write(
        "This dashboard uses a WHO-style country-year life expectancy dataset covering mortality, disease burden, "
        "immunization, health spending, GDP, income composition, schooling, and population health indicators."
    )
    st.write(f"Current source: {source_label}")
    st.write(f"Raw records: {len(raw_df):,}. Clean analytical records: {len(clean_df):,}.")

    st.subheader("Variable Dictionary")
    definitions = pd.DataFrame(
        [{"Variable": variable, "Definition": definition} for variable, definition in VARIABLE_DEFINITIONS.items()]
    )
    st.dataframe(definitions, width="stretch", hide_index=True)

    st.subheader("Cleaning and Missing Values")
    st.write(
        "Column names are stripped, standardized, and mapped to readable labels. Numeric fields are converted with "
        "coercion, records missing Life Expectancy are removed, numeric predictors are filled with median values, "
        "and categorical fields such as Country and Status are filled with Unknown when needed."
    )

    st.subheader("Dashboard Filters")
    st.write(
        "The sidebar filters restrict all pages by year range, country, and development status. Empty country "
        "selection means all countries are included."
    )

    st.subheader("Random Forest Model")
    st.write(
        "The predictive model uses available numeric predictors and one-hot encodes Status when present. The target "
        "is Life Expectancy. The data is split into training and test sets, and performance is reported with R-squared, "
        "RMSE, and MAE."
    )

    st.subheader("Limitations")
    st.markdown(
        """
        1. The data is country-level, not patient-level.
        2. Relationships are correlations, not proof of causation.
        3. Missing values and reporting differences between countries may affect results.
        4. Some countries may have incomplete data across years.
        """
    )

    st.subheader("Suggested Use")
    st.write(
        "Healthcare managers and policymakers can use the dashboard to identify countries or periods with poor "
        "life expectancy outcomes and investigate health system, disease, and socioeconomic factors for deeper review."
    )

    with st.expander("Raw data preview"):
        st.dataframe(raw_df.head(100), width="stretch")
    with st.expander("Cleaned data preview"):
        st.dataframe(clean_df.head(100), width="stretch")


def sidebar_controls(clean_df: pd.DataFrame):
    st.sidebar.header("Dashboard Filters")

    if st.sidebar.button("Reset filters"):
        st.session_state["country_filter"] = []
        if "Status" in clean_df.columns:
            st.session_state["status_filter"] = sorted(clean_df["Status"].dropna().unique().tolist())
        if "Year" in clean_df.columns:
            st.session_state["year_filter"] = (
                int(clean_df["Year"].min()),
                int(clean_df["Year"].max()),
            )

    if "Year" in clean_df.columns:
        min_year = int(clean_df["Year"].min())
        max_year = int(clean_df["Year"].max())
        year_range = st.sidebar.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=st.session_state.get("year_filter", (min_year, max_year)),
            key="year_filter",
        )
    else:
        year_range = None

    country_options = sorted(clean_df["Country"].dropna().unique().tolist()) if "Country" in clean_df.columns else []
    countries = st.sidebar.multiselect(
        "Countries (leave empty for all)",
        country_options,
        default=st.session_state.get("country_filter", []),
        key="country_filter",
    )

    status_options = sorted(clean_df["Status"].dropna().unique().tolist()) if "Status" in clean_df.columns else []
    statuses = st.sidebar.multiselect(
        "Development status",
        status_options,
        default=st.session_state.get("status_filter", status_options),
        key="status_filter",
    )

    map_metrics = available_columns(
        clean_df,
        [
            TARGET,
            "Adult Mortality",
            "BMI",
            "HIV/AIDS",
            "GDP",
            "Schooling",
            "Income Composition",
            "Polio",
            "Diphtheria",
            "Total Expenditure",
        ],
    )
    map_metric = st.sidebar.selectbox("Map metric", map_metrics, index=0 if map_metrics else None)

    page = st.sidebar.radio(
        "Dashboard page",
        [
            "Executive Overview",
            "Trends Over Time",
            "Mortality & Disease Burden",
            "Healthcare System Indicators",
            "Socioeconomic Drivers",
            "Global Map",
            "Predictive Model",
            "Data Dictionary & Methodology",
        ],
    )

    return year_range, countries, statuses, map_metric, page


def main():
    if not password_gate():
        return

    st.title("Life Expectancy Analytics Dashboard")
    st.caption("Predicting population health outcomes using mortality, immunization, and socioeconomic indicators")

    uploaded_file = st.sidebar.file_uploader("Upload Life Expectancy dataset", type=["csv", "xlsx", "xls"])
    raw_df, source_label = read_uploaded_or_default(uploaded_file)

    if raw_df is None:
        st.info("Please upload the Life Expectancy dataset to begin.")
        return

    clean_df = clean_data(raw_df)
    if clean_df.empty:
        st.error("The dataset could not be cleaned because a Life Expectancy column was not found.")
        st.stop()

    st.sidebar.success(source_label)

    with st.expander("Dataset preview"):
        st.dataframe(clean_df.head(25), width="stretch")

    year_range, countries, statuses, map_metric, page = sidebar_controls(clean_df)
    filtered_df = apply_filters(clean_df, year_range, countries, statuses)

    st.markdown(
        f"<p class='small-muted'>Showing {len(filtered_df):,} records from {clean_df['Country'].nunique() if 'Country' in clean_df.columns else 'available'} countries after filters.</p>",
        unsafe_allow_html=True,
    )

    if page == "Executive Overview":
        executive_overview(filtered_df)
    elif page == "Trends Over Time":
        trends_over_time(filtered_df)
    elif page == "Mortality & Disease Burden":
        mortality_page(filtered_df)
    elif page == "Healthcare System Indicators":
        healthcare_page(filtered_df)
    elif page == "Socioeconomic Drivers":
        socioeconomic_page(filtered_df)
    elif page == "Global Map":
        global_map_page(filtered_df, map_metric)
    elif page == "Predictive Model":
        predictive_model_page(clean_df)
    elif page == "Data Dictionary & Methodology":
        methodology_page(raw_df, clean_df, source_label)


if __name__ == "__main__":
    main()
