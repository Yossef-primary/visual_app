"""
A Needle in the Kindle — Premium Data Journalism Dashboard
============================================================
Interactive exploration of the digital publishing revolution (2000-2022).

Sections:
    - Executive Summary
    - The Economic Divide (H1)
    - The Loss Leader Identity (H2)
    - Price as a Digital Fingerprint (H3 - ML Predictor)
    - Reading Communities (Network Analysis - pending real data)
    - Volume Over Time (animated bar chart race)

NOTE ON DATA: This app is built to run against the full production dataset
(2000-2022, both Books and Kindle_Store sources, balanced Is_Kindle labels).
If a smaller/partial sample is loaded instead, the app degrades gracefully
(no crashes) and surfaces clear notices instead of misleading charts.
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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="A Needle in the Kindle",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM CSS THEME
# ============================================================
PALETTE = {
    "navy": "#14213D",          # Physical / print books
    "navy_light": "#2A3B63",
    "kindle_orange": "#FF9900",  # Digital / Kindle
    "orange_light": "#FFB84D",
    "bg": "#FAFAFA",
    "card_bg": "#FFFFFF",
    "text_dark": "#1F2933",
    "text_muted": "#64748B",
    "border": "#E5E7EB",
    "accent_purple": "#4B0082",
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Hide default Streamlit chrome (menu, footer, deploy button) */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stDeployButton"] {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ visibility: hidden; }}

    /* Base */
    html, body, .main {{
        background-color: {PALETTE['bg']};
        font-family: 'Inter', -apple-system, sans-serif;
        color: {PALETTE['text_dark']};
    }}

    h1, h2, h3 {{
        font-family: 'Playfair Display', Georgia, serif;
        color: {PALETTE['navy']};
        letter-spacing: -0.01em;
    }}

    h1 {{ font-weight: 800; }}
    h2 {{ font-weight: 700; border-bottom: 2px solid {PALETTE['border']}; padding-bottom: 0.4rem; }}

    p, li, span, label {{ color: {PALETTE['text_muted']}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {PALETTE['navy']};
        border-right: 1px solid {PALETTE['border']};
    }}
    [data-testid="stSidebar"] * {{ color: #F5F5F5 !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
        font-family: 'Playfair Display', serif;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {PALETTE['card_bg']};
        border: 1px solid {PALETTE['border']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricValue"] {{
        color: {PALETTE['navy']};
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {PALETTE['text_muted']};
        font-weight: 500;
    }}
    div[data-testid="stMetricDelta"] {{ color: {PALETTE['kindle_orange']}; }}

    /* Sliders */
    .stSlider > div > div > div > div {{ background-color: {PALETTE['kindle_orange']} !important; }}

    /* Expanders */
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {PALETTE['navy']};
        background-color: {PALETTE['card_bg']};
        border-radius: 8px;
    }}

    /* Info / notice boxes */
    .pending-box {{
        background-color: #FFF7EC;
        border: 1px dashed {PALETTE['kindle_orange']};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {PALETTE['text_dark']};
        font-size: 0.95rem;
    }}
    .story-box {{
        background-color: {PALETTE['card_bg']};
        border-left: 4px solid {PALETTE['navy']};
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.8rem;
    }}
    .story-box.orange {{ border-left-color: {PALETTE['kindle_orange']}; }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def clean_price(val):
    """Robust parsing of price strings to extract numeric values.
    Handles values like 'from 36.34', '-', or missing entries gracefully."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ("", "none", "nan", "n/a", "null", "—"):
        return np.nan
    s_clean = s.replace(",", "")
    matches = re.findall(r"\d+\.?\d*", s_clean)
    if not matches:
        return np.nan
    try:
        price = float(matches[0])
        if price <= 0 or not np.isfinite(price):
            return np.nan
        return price
    except ValueError:
        return np.nan


@st.cache_data
def load_data():
    """Loads and cleans the dataset. Caches the result across sessions.
    Returns an empty DataFrame (rather than crashing) if the file is missing
    or required columns are absent, so the rest of the app can show a
    friendly notice instead of throwing an unhandled exception."""
    try:
        df = pd.read_csv("classified_books.csv")
    except FileNotFoundError:
        st.error("Data file 'classified_books.csv' not found in the app directory.")
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
        st.warning(f"Missing expected columns: {missing_cols}. Some sections may be limited.")
        for c in missing_cols:
            df[c] = np.nan

    df = df.dropna(subset=[c for c in required if c in df.columns])

    if not df.empty:
        df["year"] = df["year"].astype(int)

    # Derive a source/format label if not present, for volume charts
    if "source_db" not in df.columns:
        if "true_format" in df.columns:
            df["source_db"] = df["true_format"].apply(
                lambda x: "Kindle_Store" if str(x).strip().lower() == "kindle" else "Books"
            )
        else:
            df["source_db"] = df["Is_Kindle"].apply(lambda x: "Kindle_Store" if x == 1 else "Books")

    if "category" not in df.columns:
        df["category"] = "Unknown"

    return df


df = load_data()

if df.empty:
    st.error("No usable data available. Please check that 'classified_books.csv' is present and correctly formatted.")
    st.stop()

N_YEARS = df["year"].nunique()
N_SOURCES = df["source_db"].nunique()
N_CLASSES_KINDLE = df["Is_Kindle"].nunique()

# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource
def train_rf_model(data):
    """Trains the Random Forest classifier. Returns (model, features, ok_flag).
    ok_flag is False if the target has fewer than 2 classes, since scikit-learn
    cannot fit a classifier on a single class — this avoids a hard crash."""
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
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 📚 A Needle in the Kindle")
st.sidebar.caption("The economics of the digital publishing revolution, 2000–2022")
st.sidebar.divider()

page = st.sidebar.radio(
    "Explore the story",
    [
        "🏠 Executive Summary",
        "💰 The Economic Divide",
        "🎯 The Loss Leader Identity",
        "🤖 Price as a Digital Fingerprint",
        "🕸️ Reading Communities",
        "📈 Volume Over Time",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Dataset in memory: {len(df):,} titles · {N_YEARS} year(s) · {N_SOURCES} source(s)")
if N_YEARS <= 1 or N_SOURCES <= 1:
    st.sidebar.warning(
        "The loaded sample spans a limited range of years/sources. "
        "Time-series and comparison views will be simplified until the full dataset is used."
    )

# ============================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================
if page == "🏠 Executive Summary":
    st.title("📚 A Needle in the Kindle")
    st.markdown(
        "How Amazon's Kindle Direct Publishing (KDP) reshaped the economics, "
        "pricing, and reading behavior of the book market."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles analyzed", f"{len(df):,}")
    c2.metric("Kindle share", f"{(df['Is_Kindle'].mean() * 100):.1f}%" if N_CLASSES_KINDLE > 1 else "n/a")
    c3.metric("Median price (2022 $)", f"${df['price_real_2022'].median():.2f}")
    c4.metric("Years covered", f"{df['year'].min()}–{df['year'].max()}")

    st.markdown("### The three hypotheses this dashboard tests")
    st.markdown("""
    <div class="story-box"><b>H1 — The Economic Divide:</b> The digital revolution triggered an
    explosion in publication volume, driving digital book prices toward a floor and creating a
    persistent pricing gap between physical and digital formats.</div>
    <div class="story-box orange"><b>H2 — The "Loss Leader" Identity:</b> A distinct low-price,
    high-engagement segment emerged. Is it driven purely by Indie authors chasing exposure — or
    something else?</div>
    <div class="story-box"><b>H3 — Price as a Digital Fingerprint:</b> A book's price predicts its
    publishing platform (Kindle vs. Physical) more strongly than its publication year or reader
    engagement.</div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 A fourth thread — **Reading Communities & Network Analysis (PageRank)** — is in "
        "progress. See the dedicated section in the sidebar for the current status."
    )

# ============================================================
# PAGE: THE ECONOMIC DIVIDE (H1)
# ============================================================
elif page == "💰 The Economic Divide":
    st.header("💰 The Economic Divide")
    st.markdown(
        "KDP's 2007 launch and Kindle Unlimited's 2014 debut are the two hinge points of this "
        "story. Together they crashed digital prices toward a floor of roughly **$6**, while "
        "physical books held a comparatively stable **$13–$16** median."
    )

    with st.expander("📖 Read the full explanation"):
        st.markdown("""
        Once anyone could self-publish to the Kindle Store, digital supply became effectively
        unlimited. That supply shock pushed digital prices down far faster than print prices,
        which are constrained by physical production and distribution costs. A Spearman rank
        correlation on the full research dataset found both trends statistically significant
        (p < 0.001), with the digital market's decline (rho = -0.363) markedly steeper than
        print's (rho = -0.231).
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Publication volume")
        vol = df.groupby(["year", "source_db"]).size().reset_index(name="Titles")
        if N_YEARS > 1:
            fig_vol = px.line(
                vol, x="year", y="Titles", color="source_db", markers=True,
                color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
                labels={"year": "Publication Year", "source_db": "Format"},
            )
        else:
            fig_vol = px.bar(
                vol, x="source_db", y="Titles", color="source_db",
                color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
                labels={"source_db": "Format"},
            )
            st.caption("Only one publication year is present in this sample — showing a snapshot instead of a trend.")
        fig_vol.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_vol, width='stretch')

    with col2:
        st.subheader("Median real price (2022 USD)")
        price_trend = df.groupby(["year", "source_db"])["price_real_2022"].median().reset_index()
        if N_YEARS > 1:
            fig_price = px.line(
                price_trend, x="year", y="price_real_2022", color="source_db", markers=True,
                color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
                labels={"year": "Publication Year", "price_real_2022": "Median Price ($)", "source_db": "Format"},
            )
        else:
            fig_price = px.bar(
                price_trend, x="source_db", y="price_real_2022", color="source_db",
                color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
                labels={"source_db": "Format", "price_real_2022": "Median Price ($)"},
            )
            st.caption("Only one publication year is present in this sample — showing a snapshot instead of a trend.")
        fig_price.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_price, width='stretch')

# ============================================================
# PAGE: THE LOSS LEADER IDENTITY (H2)
# ============================================================
elif page == "🎯 The Loss Leader Identity":
    st.header("🎯 The Loss Leader Identity")
    st.markdown(
        "An unsupervised **K-Means** pass on price and review volume isolates a cheap, "
        "high-engagement segment. The surprising finding: it's not dominated by Indie authors."
    )

    st.markdown("""
    <div class="story-box orange">
    Cross-tabulating the low-price / high-engagement cluster against publisher category shows it
    is <b>74.8% Traditional publishers</b> — not Self-Published authors as initially hypothesized.
    Legacy publishers appear to be aggressively discounting digital backlists to compete for
    reader attention, rather than Indie authors being the primary driver of the "loss leader"
    strategy.
    </div>
    """, unsafe_allow_html=True)

    n_available = len(df)
    if n_available < 20:
        st.warning(
            f"Only {n_available} rows are available after cleaning — too few for a stable "
            "K-Means fit. Showing a scatter of the raw data instead of cluster assignments."
        )
        fig_scatter = px.scatter(
            df, x="price_real_2022", y="rating_number", color="category",
            hover_data=["year", "Is_Kindle"], log_y=True,
            labels={"price_real_2022": "Real Price (2022 USD)", "rating_number": "Total Ratings"},
            opacity=0.75,
        )
    else:
        sample_df = df.sample(n=min(5000, n_available), random_state=42).copy()
        features_km = ["price_real_2022", "rating_number"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_df[features_km])

        k = min(4, sample_df.shape[0])
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        sample_df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

        fig_scatter = px.scatter(
            sample_df, x="price_real_2022", y="rating_number", color="Cluster",
            hover_data=["year", "Is_Kindle", "category"], log_y=True,
            title="Price vs. Engagement (Log Scale)",
            labels={"price_real_2022": "Real Price (2022 USD)", "rating_number": "Total Ratings"},
            opacity=0.75,
            color_discrete_sequence=[PALETTE["navy"], PALETTE["kindle_orange"], PALETTE["accent_purple"], "#8FA6C9"],
        )

        # Identify the "loss leader" cluster: lowest median price
        cluster_price = sample_df.groupby("Cluster")["price_real_2022"].median()
        loss_leader_cluster = cluster_price.idxmin()
        ll_df = sample_df[sample_df["Cluster"] == loss_leader_cluster]
        cat_share = (ll_df["category"].value_counts(normalize=True) * 100).round(1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Loss Leader cluster size", f"{len(ll_df):,} titles")
        c2.metric("Median price in cluster", f"${ll_df['price_real_2022'].median():.2f}")
        if len(cat_share) > 0:
            top_cat = cat_share.index[0]
            c3.metric(f"Top category: {top_cat}", f"{cat_share.iloc[0]}%")

    fig_scatter.update_hovertemplate = None
    fig_scatter.update_traces(
        hovertemplate="Price: $%{x:.2f}<br>Ratings: %{y:,.0f}<extra></extra>"
    )
    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("#### Understanding the segments")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**The Loss Leader:** High engagement, near-zero price — legacy publishers discounting backlist titles to compete, alongside Indie authors giving away series openers.")
        st.markdown("**Premium Print:** High price, moderate engagement — typically academic textbooks or niche non-fiction.")
    with col_b:
        st.markdown("**The Sweet Spot:** Mid-tier pricing ($9–$14) with consistent reviews — standard commercial fiction.")
        st.markdown("**The Long Tail:** Low price, low engagement — independent titles still finding an audience.")

# ============================================================
# PAGE: PRICE AS A DIGITAL FINGERPRINT (H3 - ML PREDICTOR)
# ============================================================
elif page == "🤖 Price as a Digital Fingerprint":
    st.header("🤖 Price as a Digital Fingerprint")
    st.markdown(
        "A Random Forest classifier trained on price, publication year, and engagement metrics "
        "tests whether economics alone can predict a book's platform."
    )

    with st.expander("📖 What the research found"):
        st.markdown("""
        On the full research dataset, the model reached **81.6% accuracy** against a **69.2%**
        majority-class baseline. Feature importance confirmed the hypothesis: **Real Price**
        (Gini importance 0.569) was by far the strongest predictor, ahead of publication year
        (0.332), with reader engagement metrics contributing very little. A notable limitation:
        the model shows lower recall for the Kindle class, since a subset of premium-priced
        Kindle titles ($15–$20) economically resemble physical books.
        """)

    if not rf_ok:
        st.markdown(f"""
        <div class="pending-box">
        ⚠️ <b>Live predictor unavailable with the current dataset.</b><br>
        The loaded sample only contains a single class in the <code>Is_Kindle</code> label
        column, so a classifier cannot be trained (scikit-learn requires at least two classes).
        This will resolve automatically once the full, balanced dataset (both Books and
        Kindle_Store labels) is loaded.
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            price = st.slider("Real Price (2022 USD)", 0.0, 50.0, 15.0, 0.5)
            year = st.slider(
                "Publication Year",
                int(df["year"].min()), max(int(df["year"].max()), int(df["year"].min()) + 1),
                int(df["year"].median()), 1,
            )
        with col2:
            reviews = st.number_input("Number of Ratings", min_value=1, max_value=100000, value=250, step=10)
            rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)

        input_data = pd.DataFrame([[price, year, reviews, rating]], columns=rf_features)
        prediction = rf_model.predict(input_data)[0]
        prob = rf_model.predict_proba(input_data)[0]
        kindle_prob = prob[1] * 100 if len(prob) > 1 else prob[0] * 100

        st.divider()
        gauge_col, text_col = st.columns([1, 1])

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=kindle_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Probability of Kindle Release (%)", "font": {"size": 18}},
                number={"suffix": "%", "font": {"color": PALETTE["navy"]}},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": PALETTE["text_muted"]},
                    "bar": {"color": PALETTE["kindle_orange"]},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": PALETTE["border"],
                    "steps": [
                        {"range": [0, 50], "color": "#F1F3F5"},
                        {"range": [50, 100], "color": "#FFF1DB"},
                    ],
                    "threshold": {"line": {"color": PALETTE["navy"], "width": 3}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, width='stretch')

        with text_col:
            st.subheader("Model classification")
            if prediction == 1:
                st.success("📱 **CLASSIFIED AS: KINDLE STORE**")
                st.markdown("This price, recency, and engagement profile is highly characteristic of a digital release.")
            else:
                st.info("📖 **CLASSIFIED AS: PHYSICAL BOOK**")
                st.markdown("This profile aligns more closely with traditional print publishing economics.")

            with st.expander("How does the Random Forest decide?"):
                st.write("""
                The model uses a collection of decision trees that look at historical price
                thresholds. Extremely low prices combined with high review volume frequently
                trigger the "Kindle" classification, consistent with independent authors using
                low-cost volume strategies.
                """)

        # Feature importance chart
        st.subheader("What drives the prediction?")
        importances = pd.DataFrame({
            "Feature": rf_features,
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            importances, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=[PALETTE["navy_light"], PALETTE["kindle_orange"]],
        )
        fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_imp, width='stretch')

# ============================================================
# PAGE: READING COMMUNITIES (NETWORK ANALYSIS / PAGERANK)
# ============================================================
elif page == "🕸️ Reading Communities":
    st.header("🕸️ Reading Communities & Network Analysis")

    st.markdown("""
    <div class="pending-box">
    ⚠️ <b>Analysis pending — not yet computed from real data.</b><br>
    The project notes reference a planned PageRank / community-detection study of reader
    behavior (e.g. an "Indie centrality surge post-2014" and a distinct "binge-reading"
    reader community), but no network/graph data or computed results were available in the
    provided files to back specific figures. Rather than display invented numbers, this
    section documents the intended methodology so real results can be dropped in once the
    underlying co-review / co-purchase graph and PageRank scores are computed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Intended methodology")
    st.markdown("""
    1. **Build a reader-book bipartite graph** from review activity (e.g. edges between books
       reviewed by the same reader, or reader-to-book edges directly).
    2. **Run community detection** (e.g. Louvain or label propagation) to identify clusters of
       readers/books that are more densely connected internally than to the rest of the graph.
    3. **Compute PageRank** per book (or per author/publisher) within each era to measure
       centrality — i.e., how influential a title is within the reading network, not just how
       many ratings it has.
    4. **Compare centrality trends over time** (pre- vs. post-Kindle Unlimited, 2014) and compare
       genre/format composition across the detected communities.
    """)

    st.markdown("### What to look for once data is available")
    st.markdown("""
    - Whether Indie-published titles' network centrality changes materially after 2014.
    - Whether distinct, largely non-overlapping reader communities emerge (e.g. a genre-driven
      "binge-reading" community vs. readers of classic/literary fiction).
    - Whether format (Kindle vs. print) correlates with community membership independent of price.
    """)

    st.caption(
        "This card is a structural placeholder — swap in real PageRank scores and community "
        "labels here (e.g. a node-link Plotly figure or a stacked composition chart) once the "
        "graph analysis is complete."
    )

# ============================================================
# PAGE: VOLUME OVER TIME (ANIMATED BAR CHART RACE)
# ============================================================
elif page == "📈 Volume Over Time":
    st.header("📈 The Digital Publishing Explosion")

    if N_YEARS <= 1:
        st.markdown(f"""
        <div class="pending-box">
        ⚠️ <b>Animation unavailable.</b><br>
        The loaded sample only covers a single publication year ({int(df['year'].iloc[0])}),
        so an animated time-series race can't be built. Showing a static snapshot instead.
        This will animate correctly once the full multi-year dataset (2000–2022) is loaded.
        </div>
        """, unsafe_allow_html=True)

        snapshot = df.groupby("source_db").size().reset_index(name="Titles")
        fig_static = px.bar(
            snapshot, x="Titles", y="source_db", color="source_db", orientation="h",
            color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
            labels={"source_db": "Format"},
        )
        fig_static.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_static, width='stretch')
    else:
        years = range(int(df["year"].min()), int(df["year"].max()) + 1)
        sources = df["source_db"].unique()
        idx = pd.MultiIndex.from_product([years, sources], names=["year", "source_db"])

        annual_df = (
            df.groupby(["year", "source_db"]).size()
            .reindex(idx, fill_value=0).reset_index(name="Annual_Count")
            .sort_values(by="year")
        )
        annual_df["Cumulative_Volume"] = annual_df.groupby("source_db")["Annual_Count"].cumsum()

        fig_race = px.bar(
            annual_df, x="Cumulative_Volume", y="source_db", color="source_db",
            animation_frame="year", animation_group="source_db", orientation="h",
            range_x=[0, annual_df["Cumulative_Volume"].max() * 1.05],
            title="Cumulative Titles Published Over Time",
            labels={"Cumulative_Volume": "Total Books Published", "source_db": "Format", "year": "Year"},
            color_discrete_map={"Books": PALETTE["navy"], "Kindle_Store": PALETTE["kindle_orange"]},
        )

        if fig_race.layout.updatemenus:
            fig_race.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

        fig_race.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"},
            height=420,
        )
        st.plotly_chart(fig_race, width='stretch')
        st.caption("Press 'Play' on the timeline axis above to watch the Kindle store rapidly overtake traditional books.")
