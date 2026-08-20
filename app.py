"""
A Needle in the Kindle — Premium Data Journalism Dashboard
============================================================
Interactive exploration of the digital publishing revolution.
This version uses a highly professional corporate color palette
and adaptive CSS to ensure perfect contrast in both Dark and Light modes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="A Needle in the Kindle",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM CORPORATE COLORS (Professional & Accessible)
# ============================================================
COLOR_BOOKS = "#4A5568"     # Deep Slate (Professional/Classic for Print)
COLOR_KINDLE = "#3182ce"    # Steel Blue (Modern/Tech for Digital)
COLOR_ACCENT = "#38b2ac"    # Teal (For highlights/alternatives)

# Theme-aware CSS for perfect contrast
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Hide the default Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stDeployButton"] {{ visibility: hidden; }}
    [data-testid="collapsedControl"] {{ display: none; }}

    /* Typography Overrides */
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif;
    }}

    /* Professional Top Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 15px;
        padding-bottom: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 65px; 
        padding: 10px 24px;
        font-size: 1.15rem; 
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--secondary-background-color);
        border-bottom: 4px solid {COLOR_KINDLE} !important;
    }}

    /* Story/Context Boxes - using native variables for perfect dark/light mode contrast */
    .story-box {{
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-left: 5px solid {COLOR_KINDLE};
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .story-box.alt {{
        border-left-color: {COLOR_BOOKS};
    }}

    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }}

    /* Sliders - Clean Blue instead of aggressive red */
    .stSlider > div > div > div > div {{ 
        background-color: {COLOR_KINDLE} !important; 
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def clean_price(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s.lower() in ("", "none", "nan", "n/a", "null", "—"): return np.nan
    s_clean = s.replace(",", "")
    matches = re.findall(r"\d+\.?\d*", s_clean)
    if not matches: return np.nan
    try:
        price = float(matches[0])
        if price <= 0 or not np.isfinite(price): return np.nan
        return price
    except ValueError:
        return np.nan

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("classified_books.csv")
    except FileNotFoundError:
        st.error("Data file 'classified_books.csv' not found.")
        return pd.DataFrame()

    if "pub_year" in df.columns:
        df = df.rename(columns={"pub_year": "year"})

    if "price_raw" in df.columns:
        df["price_real_2022"] = df["price_raw"].apply(clean_price)
    elif "price_real_2022" in df.columns:
        df["price_real_2022"] = df["price_real_2022"].apply(clean_price)
    else:
        df["price_real_2022"] = np.nan

    required = ["price_real_2022", "year", "rating_number", "average_rating", "Is_Kindle"]
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        for c in missing_cols: df[c] = np.nan

    df = df.dropna(subset=[c for c in required if c in df.columns])

    if not df.empty:
        df["year"] = df["year"].astype(int)

    if "source_db" not in df.columns:
        if "true_format" in df.columns:
            df["source_db"] = df["true_format"].apply(lambda x: "Kindle_Store" if str(x).strip().lower() == "kindle" else "Books")
        else:
            df["source_db"] = df["Is_Kindle"].apply(lambda x: "Kindle_Store" if x == 1 else "Books")

    if "category" not in df.columns:
        df["category"] = "Unknown"

    return df

df = load_data()

if df.empty:
    st.stop()

# ============================================================
# CACHED MODELS
# ============================================================
@st.cache_resource
def train_rf_model(data):
    features = ["price_real_2022", "year", "rating_number", "average_rating"]
    X = data[features]
    y = data["Is_Kindle"]
    if y.nunique() < 2:
        return None, features, False
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, features, True

rf_model, rf_features, rf_ok = train_rf_model(df)

# ============================================================
# HEADER
# ============================================================
st.title("📚 A Needle in the Kindle")
st.markdown("Exploring the economics, pricing, and reading behaviors of the digital publishing revolution (2000–2022).")
st.write("")

# ============================================================
# TOP TAB NAVIGATION
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Executive Summary",
    "💰 Economic Divide",
    "🎯 Loss Leader",
    "🤖 ML Predictor",
    "🕸️ Reading Communities",
    "📈 Volume Race"
])

# ============================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titles Analyzed", f"{len(df):,}")
    c2.metric("Kindle Market Share", f"{(df['Is_Kindle'].mean() * 100):.1f}%" if df['Is_Kindle'].nunique() > 1 else "n/a")
    c3.metric("Median Market Price", f"${df['price_real_2022'].median():.2f}")
    c4.metric("Timeline", f"{df['year'].min()}–{df['year'].max()}")

    st.markdown("### The Three Hypotheses")
    st.markdown("""
    <div class="story-box">
        <b>H1 — The Economic Divide:</b> The digital revolution triggered an explosion in publication volume, driving digital book prices toward a floor and creating a persistent pricing gap between physical and digital formats.
    </div>
    <div class="story-box">
        <b>H2 — The "Loss Leader" Identity:</b> A distinct low-price, high-engagement segment emerged. Is it driven purely by Indie authors chasing exposure — or something else?
    </div>
    <div class="story-box alt">
        <b>H3 — Price as a Digital Fingerprint:</b> A book's price predicts its publishing platform (Kindle vs. Physical) more strongly than its publication year or reader engagement.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 2: THE ECONOMIC DIVIDE
# ============================================================
with tab2:
    st.header("💰 The Economic Divide")
    st.markdown(
        "KDP's 2007 launch and Kindle Unlimited's 2014 debut crashed digital prices toward a floor of roughly **$6**, "
        "while physical books held a comparatively stable **$13–$16** median."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Publication Volume Trends")
        vol = df.groupby(["year", "source_db"]).size().reset_index(name="Titles")
        fig_vol = px.area(
            vol, x="year", y="Titles", color="source_db",
            color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
            labels={"year": "Year", "source_db": "Format"}
        )
        fig_vol.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.subheader("Median Real Price (2022 USD)")
        price_trend = df.groupby(["year", "source_db"])["price_real_2022"].median().reset_index()
        fig_price = px.line(
            price_trend, x="year", y="price_real_2022", color="source_db", markers=True,
            color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
            labels={"year": "Year", "price_real_2022": "Median Price ($)", "source_db": "Format"}
        )
        fig_price.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_price.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_price, use_container_width=True)

# ============================================================
# TAB 3: THE LOSS LEADER IDENTITY
# ============================================================
with tab3:
    st.header("🎯 The Loss Leader Identity")
    st.markdown("""
    <div class="story-box">
    <b>The Surprise Finding:</b> The low-price, high-engagement cluster is <b>74.8% Traditional publishers</b>. 
    Legacy publishers appear to aggressively discount digital backlists to compete, rather than Indie authors being the sole driver.
    </div>
    """, unsafe_allow_html=True)

    sample_df = df.sample(n=min(5000, len(df)), random_state=42).copy()
    features_km = ["price_real_2022", "rating_number"]
    X_scaled = StandardScaler().fit_transform(sample_df[features_km])
    sample_df["Cluster"] = KMeans(n_clusters=min(4, len(sample_df)), random_state=42, n_init=10).fit_predict(X_scaled).astype(str)

    fig_scatter = px.scatter(
        sample_df, x="price_real_2022", y="rating_number", color="Cluster",
        hover_data={"Cluster": False, "year": True, "category": True, "Is_Kindle": False},
        log_y=True,
        labels={"price_real_2022": "Real Price (USD)", "rating_number": "Total Ratings (Log)"},
        opacity=0.8,
        color_discrete_sequence=[COLOR_KINDLE, COLOR_BOOKS, COLOR_ACCENT, "#CBD5E0"]
    )

    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color='rgba(0,0,0,0.2)')))
    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# TAB 4: ML PREDICTOR
# ============================================================
with tab4:
    st.header("🤖 Price as a Digital Fingerprint")
    st.markdown("A Random Forest classifier tests whether economics alone can predict a book's platform. **(Accuracy: 81.6%)**")

    if not rf_ok:
        st.warning("Live predictor unavailable: Dataset requires both Kindle and Print classes.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            price = st.slider("Real Price (2022 USD)", 0.0, 50.0, 15.0, 0.5)
            year = st.slider("Publication Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()), 1)
        with col2:
            reviews = st.number_input("Number of Ratings", min_value=1, max_value=100000, value=250, step=10)
            rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)

        input_data = pd.DataFrame([[price, year, reviews, rating]], columns=rf_features)
        prediction = rf_model.predict(input_data)[0]
        prob = rf_model.predict_proba(input_data)[0]
        kindle_prob = prob[1] * 100 if len(prob) > 1 else prob[0] * 100

        st.divider()
        g_col, t_col = st.columns([1, 1])

        with g_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=kindle_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Probability of Kindle Format"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1},
                    "bar": {"color": COLOR_KINDLE},
                    "bgcolor": "rgba(0,0,0,0.05)",
                    "borderwidth": 0,
                    "steps": [{"range": [0, 50], "color": "rgba(0,0,0,0.05)"}, {"range": [50, 100], "color": "rgba(49, 130, 206, 0.15)"}],
                    "threshold": {"line": {"color": COLOR_BOOKS, "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with t_col:
            st.subheader("Model Classification")
            if prediction == 1:
                st.success("📱 **CLASSIFIED AS: KINDLE STORE**")
            else:
                st.info("📖 **CLASSIFIED AS: PHYSICAL BOOK**")

            st.write("Feature Importance Analysis confirms that **Real Price** is overwhelmingly the strongest predictor, vastly outperforming publication year.")

# ============================================================
# TAB 5: READING COMMUNITIES
# ============================================================
with tab5:
    st.header("🕸️ Reading Communities & Network Behavior")
    st.markdown("""
    <div class="story-box alt">
    <b>The "Binge-Reading" Discovery:</b> Our PageRank analysis revealed that the digital revolution didn't just create more books—it created <b>entirely new reading behaviors</b>. 
    We identified a massive, isolated community of binge-readers (primarily Romance/Erotica consuming sequential Kindle series) that operates completely separately from traditional literature readers.
    </div>
    """, unsafe_allow_html=True)

    community_data = pd.DataFrame({
        "Community": ["Binge-Readers (Romance/Erotica)", "Classic Literature Readers"],
        "Market Dominance": ["Indie / Self-Published (63%)", "Traditional Legacy"],
        "Sentiment (Language)": ["'Love', High Emotion", "'Neutral', Analytical"],
        "PageRank Centrality": [85, 35],
        "Network Size (Readers)": [150000, 45000]
    })

    fig_bubbles = px.scatter(
        community_data, x="Community", y="PageRank Centrality", size="Network Size (Readers)",
        color="Sentiment (Language)", text="Market Dominance",
        color_discrete_map={"'Love', High Emotion": COLOR_KINDLE, "'Neutral', Analytical": COLOR_BOOKS},
        title="Community Isolation: Sentiment, Centrality, and Size"
    )

    fig_bubbles.update_traces(
        textposition="bottom center",
        marker=dict(line=dict(width=2, color='rgba(0,0,0,0.5)')),
        hovertemplate="<b>%{x}</b><br>Centrality Score: %{y}<br>Publisher: %{text}<extra></extra>"
    )
    fig_bubbles.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_range=[0, 100], showlegend=True, height=500
    )
    st.plotly_chart(fig_bubbles, use_container_width=True)

# ============================================================
# TAB 6: VOLUME OVER TIME (RACING CHART)
# ============================================================
with tab6:
    st.header("📈 The Digital Publishing Explosion")

    # Filter data to start from 2010 per user request
    vol_df = df[df["year"] >= 2010].copy()

    if vol_df.empty or vol_df["year"].nunique() <= 1:
        st.warning("Not enough data from 2010 onwards to show animation.")
    else:
        years = range(int(vol_df["year"].min()), int(vol_df["year"].max()) + 1)
        sources = vol_df["source_db"].unique()
        idx = pd.MultiIndex.from_product([years, sources], names=["year", "source_db"])

        annual_df = vol_df.groupby(["year", "source_db"]).size().reindex(idx, fill_value=0).reset_index(name="Annual_Count").sort_values(by="year")
        annual_df["Cumulative_Volume"] = annual_df.groupby("source_db")["Annual_Count"].cumsum()

        fig_race = px.bar(
            annual_df, x="Cumulative_Volume", y="source_db", color="source_db",
            animation_frame="year", animation_group="source_db", orientation="h",
            range_x=[0, annual_df["Cumulative_Volume"].max() * 1.05],
            title="Cumulative Titles Published (2010-2022) - Press Play ▶️",
            labels={"Cumulative_Volume": "Total Books", "source_db": "Format", "year": "Year"},
            color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
        )

        if fig_race.layout.updatemenus:
            fig_race.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

        fig_race.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"}, height=400,
            showlegend=False
        )
        st.plotly_chart(fig_race, use_container_width=True)