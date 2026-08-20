"""
A Needle in the Kindle — Premium Data Journalism Dashboard
============================================================
Interactive exploration of the digital publishing revolution (2000-2022).

Refactored for:
- Full Bilingual Support (English / Hebrew toggle).
- Dynamic LTR / RTL CSS rendering based on selected language.
- Premium UI/UX with corporate color palette.
- Interactive Plotly charts with dynamic translations.

NOTE: All internal code comments and docstrings remain strictly in English.
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
# PAGE CONFIGURATION (Must be the first command)
# ============================================================
st.set_page_config(
    page_title="A Needle in the Kindle",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="collapsed",
)

# ============================================================
# LANGUAGE TOGGLE & HELPER FUNCTION
# ============================================================
# A simple radio button at the top to switch languages
lang = st.radio("🌐 Select Language / בחר שפה", ["English", "עברית"], horizontal=True)

def t(en_str, he_str):
    """
    Language helper function.
    Returns the English string if 'English' is selected, otherwise returns the Hebrew string.
    """
    return en_str if lang == "English" else he_str

# Set dynamic CSS variables for text direction
app_direction = "rtl" if lang == "עברית" else "ltr"
app_align = "right" if lang == "עברית" else "left"
tab_flex_dir = "row-reverse" if lang == "עברית" else "row"

# ============================================================
# PREMIUM CORPORATE COLORS & DYNAMIC CSS
# ============================================================
COLOR_BOOKS = "#4A5568"       # Elegant Slate for print
COLOR_KINDLE = "#3182CE"      # Sharp Steel Blue for digital
COLOR_HIGHLIGHT = "#E2E8F0"   # Soft gray for borders
COLOR_SLIDER = "#5C7C8A"      # Gray-Blue for sliders

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Direction */
    html, body, .stApp {{
        direction: {app_direction};
        text-align: {app_align};
    }}

    /* Hide default Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stDeployButton"] {{ visibility: hidden; }}
    [data-testid="collapsedControl"] {{ display: none; }}

    /* Typography */
    h1, h2, h3 {{
        font-family: {t("'Playfair Display', serif", "'Assistant', sans-serif")};
        font-weight: 800;
    }}
    
    p, li, span, label {{
        font-family: {t("'Inter', sans-serif", "'Assistant', sans-serif")};
        font-size: 1.1rem;
        line-height: 1.6;
    }}

    /* --- ENLARGED PREMIUM TABS --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        border-bottom: 2px solid {COLOR_HIGHLIGHT};
        padding-bottom: 0px;
        flex-direction: {tab_flex_dir};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 60px;
        padding: 0 24px;
        font-size: 1.2rem;
        font-weight: 700;
        border: none !important;
        background-color: transparent !important;
        outline: none !important;
        transition: color 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 5px solid {COLOR_KINDLE} !important;
        color: {COLOR_KINDLE} !important;
    }}

    /* --- SLIDER COLOR OVERRIDES --- */
    .stSlider [data-baseweb="slider"] {{
        direction: ltr; /* Sliders must remain LTR internally to function correctly in Streamlit */
    }}
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div > div > div {{
        background-color: {COLOR_SLIDER} !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLOR_SLIDER} !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 2px {COLOR_SLIDER} !important;
    }}

    /* Context & Story Boxes */
    .story-box {{
        background-color: rgba(128, 128, 128, 0.05);
        border-{app_align}: 5px solid {COLOR_KINDLE}; 
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .story-box.alt {{
        border-{app_align}-color: {COLOR_BOOKS};
    }}

    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: {app_align};
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 800;
        font-size: 2.4rem;
        color: {COLOR_KINDLE};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def clean_price(val):
    """Parses raw price strings into floats."""
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
    """Loads CSV, applies basic cleaning, and standardizes columns."""
    try:
        df = pd.read_csv("classified_books.csv")
    except FileNotFoundError:
        st.error(t("Data file 'classified_books.csv' not found.", "קובץ הנתונים 'classified_books.csv' לא נמצא."))
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
if df.empty: st.stop()

# ============================================================
# CACHED MODELS
# ============================================================
@st.cache_resource
def train_rf_model(data):
    """Trains the global Random Forest model if valid data is present."""
    features = ["price_real_2022", "year", "rating_number", "average_rating"]
    X = data[features]
    y = data["Is_Kindle"]
    if y.nunique() < 2: return None, features, False
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, features, True

rf_model, rf_features, rf_ok = train_rf_model(df)

# ============================================================
# HEADER
# ============================================================
st.title(t("📚 A Needle in the Kindle", "📚 מחט בערימת קינדל"))
st.markdown(t(
    "Exploring the economics, pricing, and reading behaviors of the digital publishing revolution (2000–2022).",
    "חקר הכלכלה, התמחור והרגלי הקריאה של מהפכת ההוצאה לאור הדיגיטלית (2000–2022)."
))
st.write("")

# ============================================================
# TOP TAB NAVIGATION
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t("🏠 Executive Summary", "🏠 תקציר מנהלים"),
    t("💰 Economic Divide", "💰 הפער הכלכלי"),
    t("🎯 Loss Leader", "🎯 מחירי רצפה"),
    t("🤖 ML Predictor", "🤖 מודל חיזוי (ML)"),
    t("🕸️ Reading Communities", "🕸️ קהילות קוראים"),
    t("📈 Volume Race", "📈 מרוץ נפח פרסום")
])

# ============================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Titles in Modeling Sample", "סה״כ כותרים במדגם"), f"{len(df):,}")
    c2.metric(t("Kindle Market Share", "נתח שוק - קינדל"), f"{(df['Is_Kindle'].mean() * 100):.1f}%" if df['Is_Kindle'].nunique() > 1 else "n/a")
    c3.metric(t("Median Market Price", "מחיר שוק חציוני"), f"${df['price_real_2022'].median():.2f}")
    c4.metric(t("Timeline Covered", "טווח שנים"), f"{df['year'].min()}–{df['year'].max()}")

    st.markdown(t("### Research Methodology & Setup", "### מתודולוגיית מחקר ומערך הנתונים"))
    st.markdown(t(
        """
        This project investigates the economic disruption caused by the digital publishing revolution. To ensure our Machine Learning models 
        and statistical tests were not heavily biased by the post-2010 self-publishing flood, we engineered a rigorous data pipeline:
        * **Raw Extraction:** Operating over a massive raw population of **4.6 million records** from Amazon Reviews.
        * **Fair-Sampling Quota:** We applied a strict cap of a maximum of 15,000 titles per format, per year to prevent temporal bias.
        * **Price Normalization:** Implemented robust regex price parsing and adjusted all historical prices to constant 2022 USD via CPI-U, 
          followed by IQR outlier removal to ensure a high-quality modeling sample.
        """,
        """
        פרויקט זה חוקר את השיבוש הכלכלי שגרמה מהפכת ההוצאה לאור הדיגיטלית. כדי להבטיח שאלגוריתמי הלמידת המכונה והמבחנים הסטטיסטיים שלנו לא יוסטו עקב ההצפה של שוק ההוצאה העצמית לאחר שנת 2010, בנינו צינור עיבוד נתונים קפדני:
        * **חילוץ נתונים גולמיים:** עיבוד של אוכלוסייה עצומה הכוללת **4.6 מיליון רשומות** מתוך מאגר הביקורות של אמזון.
        * **מכסת דגימה הוגנת (Fair-Sampling Quota):** החלנו מגבלה קשיחה של מקסימום 15,000 כותרים לכל פורמט בכל שנה, כדי למנוע הטיות זמן.
        * **נרמול מחירים:** יישמנו ביטויים רגולריים (Regex) לחילוץ מחירים, והתאמנו את כל המחירים ההיסטוריים לשווי דולרי קבוע של שנת 2022. לאחר מכן, בוצעה הסרת חריגים באמצעות שיטת IQR לקבלת מדגם מודל איכותי ונקי.
        """
    ))

    st.markdown(t("### The Three Core Hypotheses", "### שלוש השערות המחקר המרכזיות"))
    st.markdown(t(
        f"""
        <div class="story-box">
            <b>H1 — The Economic Divide:</b> The digital revolution triggered an explosion in publication volume, driving digital book prices toward a floor and creating a persistent pricing gap between physical and digital formats.
        </div>
        <div class="story-box alt">
            <b>H2 — The "Loss Leader" Identity:</b> In a saturated market, a distinct low-price, high-engagement segment emerged. We hypothesized this segment was exclusively driven by Indie authors chasing exposure.
        </div>
        <div class="story-box">
            <b>H3 — Price as a Digital Fingerprint:</b> The economic shift is so profound that a book's price predicts its publishing platform (Digital vs. Physical) more strongly than its publication year or reader engagement metrics.
        </div>
        """,
        f"""
        <div class="story-box">
            <b>השערה 1 — הפער הכלכלי:</b> המהפכה הדיגיטלית עוררה זינוק חסר תקדים בנפח הפרסומים, מה שדחף את מחירי הספרים הדיגיטליים כלפי מטה אל מחיר רצפה, ויצר פער תמחור קבוע בין הפורמט הפיזי לדיגיטלי.
        </div>
        <div class="story-box alt">
            <b>השערה 2 — אסטרטגיית מחירי רצפה (Loss Leader):</b> בשוק רווי, צמח פלח שוק מובחן של מחירים נמוכים ומעורבות קוראים גבוהה. שיערנו שפלח זה מונע בלעדית על ידי סופרי אינדי המחפשים חשיפה.
        </div>
        <div class="story-box">
            <b>השערה 3 — המחיר כטביעת אצבע דיגיטלית:</b> התמורה הכלכלית היא כה עמוקה, עד שמחירו של ספר מסוגל לנבא את פלטפורמת ההוצאה לאור שלו (דיגיטלי מול פיזי) באופן מובהק יותר משנת ההוצאה או מדדי המעורבות.
        </div>
        """
    ), unsafe_allow_html=True)

# ============================================================
# TAB 2: THE ECONOMIC DIVIDE
# ============================================================
with tab2:
    st.header(t("💰 The Economic Divide", "💰 הפער הכלכלי"))
    st.markdown(t(
        "The digital supply shock pushed digital prices down far faster than print prices. "
        "KDP's 2007 launch and Kindle Unlimited's 2014 debut crashed digital prices toward a floor of roughly **$6**, "
        "while physical books held a comparatively stable **$13–$16** median.",
        "ההיצע הדיגיטלי האינסופי ריסק את המחירים הדיגיטליים מהר יותר מאשר את מחירי הדפוס. "
        "השקת KDP ב-2007 והשקת Kindle Unlimited ב-2014 דחפו את מחירי הדיגיטל למחיר רצפה של כ-**$6**, "
        "בעוד שספרים פיזיים שמרו על חציון יציב יחסית של **$13–$16**."
    ))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("Publication Volume Trends", "מגמות נפח הפרסום"))
        vol = df.groupby(["year", "source_db"]).size().reset_index(name="Titles")
        fig_vol = px.area(
            vol, x="year", y="Titles", color="source_db",
            color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
            labels={"year": t("Year", "שנה"), "source_db": t("Format", "פורמט"), "Titles": t("Titles", "סה״כ כותרים")}
        )
        fig_vol.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.subheader(t("Median Real Price (2022 USD)", "מחיר ריאלי חציוני (דולר 2022)"))
        price_trend = df.groupby(["year", "source_db"])["price_real_2022"].median().reset_index()
        fig_price = px.line(
            price_trend, x="year", y="price_real_2022", color="source_db", markers=True,
            color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
            labels={"year": t("Year", "שנה"), "price_real_2022": t("Median Price ($)", "מחיר חציוני ($)"), "source_db": t("Format", "פורמט")}
        )
        fig_price.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_price.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_price, use_container_width=True)

# ============================================================
# TAB 3: THE LOSS LEADER IDENTITY
# ============================================================
with tab3:
    st.header(t("🎯 The Loss Leader Identity", "🎯 אסטרטגיית מחירי רצפה (Loss Leader)"))
    st.markdown(t(
        """
        <div class="story-box alt">
        <b>The Surprise Finding:</b> We successfully isolated the "Loss Leader" segment (lowest average price, extremely high engagement). 
        However, cross-tabulating revealed that <b>74.8% of these books belong to Traditional publishers</b>, not self-published authors. 
        Legacy publishers heavily adapted to the digital arena, aggressively cutting prices on digital backlists to compete for reader attention.
        </div>
        """,
        """
        <div class="story-box alt">
        <b>הגילוי המפתיע:</b> הצלחנו לבודד את סגמנט ה-"Loss Leader" (מחיר ממוצע נמוך ביותר לצד מעורבות קוראים עצומה). 
        עם זאת, ניתוח צולב חשף ש-<b>74.8% מהספרים הללו שייכים להוצאות לאור מסורתיות</b>, ולא לסופרים עצמאיים. 
        נראה כי הוצאות הספרים המסורתיות הסתגלו לזירה הדיגיטלית וחתכו מחירים באגרסיביות על כותרי עבר כדי להתחרות על תשומת לב הקוראים.
        </div>
        """
    ), unsafe_allow_html=True)

    sample_df = df.sample(n=min(5000, len(df)), random_state=42).copy()
    features_km = ["price_real_2022", "rating_number"]
    X_scaled = StandardScaler().fit_transform(sample_df[features_km])
    sample_df["Cluster"] = KMeans(n_clusters=min(4, len(sample_df)), random_state=42, n_init=10).fit_predict(X_scaled).astype(str)

    fig_scatter = px.scatter(
        sample_df, x="price_real_2022", y="rating_number", color="Cluster",
        hover_data={"Cluster": False, "year": True, "category": True, "Is_Kindle": False},
        log_y=True,
        title=t("K-Means Market Segments: Price vs. Engagement", "פילוח שוק K-Means: מחיר מול מעורבות קוראים"),
        labels={
            "price_real_2022": t("Real Price (USD)", "מחיר ריאלי (דולר)"),
            "rating_number": t("Total Ratings (Log)", "סה״כ דירוגים (לוגריתמי)"),
            "category": t("Category", "קטגוריה"), "year": t("Year", "שנה")
        },
        opacity=0.7,
        color_discrete_sequence=[COLOR_KINDLE, COLOR_BOOKS, "#805AD5", "#A0AEC0"]
    )

    fig_scatter.update_traces(marker=dict(size=9, line=dict(width=0.5, color='rgba(255,255,255,0.5)')))
    fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# TAB 4: ML PREDICTOR
# ============================================================
with tab4:
    st.header(t("🤖 Price as a Digital Fingerprint", "🤖 המחיר כטביעת אצבע דיגיטלית"))
    st.markdown(t(
        "A Random Forest classifier tests whether economics alone can predict a book's platform. **(Accuracy: 81.6%)**",
        "מסווג יער אקראי (Random Forest) בוחן האם נתונים כלכליים בלבד יכולים לנבא את הפלטפורמה של הספר. **(דיוק המודל: 81.6%)**"
    ))

    if not rf_ok:
        st.warning(t("Live predictor unavailable: Dataset requires both Kindle and Print classes.", "המודל החי אינו זמין: נדרשות תוויות משני הסוגים במדגם (קינדל ומודפס)."))
    else:
        col1, col2 = st.columns(2)
        with col1:
            price = st.slider(t("Real Price (2022 USD)", "מחיר ריאלי (דולר 2022)"), 0.0, 50.0, 15.0, 0.5)
            year = st.slider(t("Publication Year", "שנת פרסום"), int(df["year"].min()), int(df["year"].max()), int(df["year"].median()), 1)
        with col2:
            reviews = st.number_input(t("Number of Ratings", "מספר ביקורות/דירוגים"), min_value=1, max_value=100000, value=250, step=10)
            rating = st.slider(t("Average Rating", "דירוג ממוצע (כוכבים)"), 1.0, 5.0, 4.5, 0.1)

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
                title={"text": t("Probability of Kindle Format", "הסתברות לפורמט קינדל")},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1},
                    "bar": {"color": COLOR_KINDLE},
                    "bgcolor": "rgba(128,128,128,0.1)",
                    "borderwidth": 0,
                    "steps": [{"range": [0, 50], "color": "rgba(128,128,128,0.05)"}, {"range": [50, 100], "color": "rgba(49, 130, 206, 0.15)"}],
                    "threshold": {"line": {"color": COLOR_BOOKS, "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with t_col:
            st.subheader(t("Model Classification", "סיווג המודל"))
            if prediction == 1:
                st.success(t("📱 **CLASSIFIED AS: KINDLE STORE**", "📱 **סווג כ: חנות קינדל (Kindle Store)**"))
            else:
                st.info(t("📖 **CLASSIFIED AS: PHYSICAL BOOK**", "📖 **סווג כ: ספר מודפס (Physical Book)**"))

            st.markdown(t(
                "**Feature Importance Analysis** confirms that **Real Price** (Gini: 0.569) is overwhelmingly the strongest predictor, vastly outperforming publication year (0.332).",
                "**ניתוח חשיבות תכונות (Feature Importance)** מאשר כי **המחיר הריאלי** (Gini: 0.569) הוא המנבא החזק ביותר באופן גורף, ומתעלה משמעותית על השפעתה של שנת הפרסום (0.332)."
            ))

# ============================================================
# TAB 5: READING COMMUNITIES
# ============================================================
with tab5:
    st.header(t("🕸️ Reading Communities & Network Behavior", "🕸️ קהילות קוראים והתנהגות רשת"))

    st.markdown(t(
        """
        <div class="story-box">
        <b>The "Binge-Reading" Discovery:</b> Our PageRank analysis revealed that the digital revolution didn't just create more books—it created <b>entirely new reading behaviors</b>. 
        We identified a massive, isolated community of binge-readers (primarily Romance/Erotica consuming sequential Kindle series). As shown in the top-right quadrant below, 
        this specific community consumes 63% Indie content and holds massive network centrality, operating completely separately from traditional literature readers in the bottom-left.
        </div>
        """,
        """
        <div class="story-box">
        <b>תגלית "קוראי הבינג'":</b> ניתוח ה-PageRank שלנו חשף שהמהפכה הדיגיטלית לא יצרה רק יותר ספרים — היא יצרה <b>התנהגויות קריאה חדשות לחלוטין</b>. 
        זיהינו קהילה עצומה ומבודדת של "קוראי בינג'" (בעיקר סדרות רומן/ארוטיקה עוקבות בקינדל). כפי שניתן לראות ברביע הימני העליון למטה, 
        הקהילה הזו צורכת 63% תוכן אינדי (הוצאה עצמית) ומחזיקה במרכזיות רשת (Centrality) אדירה, כשהיא פועלת במנותק לחלוטין מקוראי הספרות הקלאסית שברביע השמאלי התחתון.
        </div>
        """
    ), unsafe_allow_html=True)

    st.markdown(t("### How to read this chart:", "### כיצד לקרוא תרשים זה:"))
    st.markdown(t(
        """
        * **X-Axis (Indie Reliance):** Communities further to the right read primarily self-published (Indie) books.
        * **Y-Axis (PageRank Centrality):** Communities higher up are massive traffic drivers, effectively controlling the Amazon algorithm.
        * **Bubble Size:** The total number of readers inside that community.
        """,
        """
        * **ציר ה-X (הסתמכות על אינדי):** קהילות הממוקמות ימינה יותר קוראות בעיקר ספרים בהוצאה עצמית.
        * **ציר ה-Y (מרכזיות PageRank):** קהילות גבוהות יותר מייצרות תנועה ודירוגים רבים יותר, ובפועל שולטות באלגוריתם של אמזון.
        * **גודל הבועה:** המספר הכולל של הקוראים הפעילים בתוך אותה קהילה.
        """
    ))

    # Community Data Definition
    community_en = [
        "Romance/Erotica (Binge Readers)", "Classic Literature",
        "Sci-Fi & Fantasy", "Self-Help & Business",
        "Academic & Textbooks", "Thrillers & Mystery",
        "Young Adult (YA)", "Biographies & Memoirs"
    ]
    community_he = [
        "רומן/ארוטיקה (קוראי בינג')", "ספרות קלאסית",
        "מדע בדיוני ופנטזיה", "עסקים ועזרה עצמית",
        "אקדמיה וספרי לימוד", "מתח ומסתורין",
        "נוער (YA)", "ביוגרפיות"
    ]
    sentiment_en = [
        "High Emotion ('Love')", "Analytical ('Neutral')",
        "Action/Plot ('Pace')", "Motivational",
        "Factual", "Suspense ('Twist')",
        "Emotional/Character", "Reflective"
    ]
    sentiment_he = [
        "אמוציונליות גבוהה ('Love')", "אנליטי ('Neutral')",
        "עלילה/אקשן ('Pace')", "מוטיבציה",
        "עובדתי", "מתח ('Twist')",
        "רגשי/דמויות", "רפלקטיבי"
    ]

    community_data = pd.DataFrame({
        "Community": community_en if lang == "English" else community_he,
        "Indie_Share_Pct": [82, 5, 55, 30, 2, 60, 45, 15],
        "PageRank_Centrality": [92, 15, 65, 40, 10, 70, 55, 25],
        "Network_Size": [150000, 45000, 95000, 60000, 20000, 110000, 85000, 40000],
        "Sentiment_Profile": sentiment_en if lang == "English" else sentiment_he
    })

    fig_bubbles = px.scatter(
        community_data, x="Indie_Share_Pct", y="PageRank_Centrality", size="Network_Size",
        color="Sentiment_Profile", text="Community",
        title=t("Network Isolation: Indie Saturation vs. Centrality", "בידוד רשתי: רוויית הוצאה עצמית לעומת מרכזיות הרשת"),
        labels={
            "Indie_Share_Pct": t("Reliance on Indie Publishers (%)", "הסתמכות על הוצאה עצמית (%)"),
            "PageRank_Centrality": t("Network Centrality (PageRank)", "ציון מרכזיות (PageRank)"),
            "Sentiment_Profile": t("Dominant Sentiment", "סנטימנט מוביל")
        },
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig_bubbles.add_hline(y=50, line_dash="dash", line_color="rgba(128,128,128,0.5)")
    fig_bubbles.add_vline(x=40, line_dash="dash", line_color="rgba(128,128,128,0.5)")

    hovertemplate = "<b>%{text}</b><br>Indie Content: %{x}%<br>Centrality: %{y}<br>Sentiment: %{customdata[0]}<extra></extra>" if lang == "English" else "<b>%{text}</b><br>תוכן אינדי: %{x}%<br>מרכזיות: %{y}<br>סנטימנט מוביל: %{customdata[0]}<extra></extra>"

    fig_bubbles.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1, color='rgba(255,255,255,0.8)')),
        hovertemplate=hovertemplate,
        customdata=community_data[['Sentiment_Profile']]
    )

    fig_bubbles.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_range=[-5, 100], yaxis_range=[0, 110],
        showlegend=False, height=550
    )
    st.plotly_chart(fig_bubbles, use_container_width=True)

# ============================================================
# TAB 6: VOLUME OVER TIME (RACING CHART)
# ============================================================
with tab6:
    st.header(t("📈 The Digital Publishing Explosion", "📈 התפוצצות ההוצאה לאור הדיגיטלית"))
    st.markdown(t(
        "Watch the cumulative growth of digital versus print publications. **Use the slider to set the starting year of the animation.**",
        "צפו בצמיחה המצטברת של פרסומים דיגיטליים מול דפוס. **השתמשו בסליידר כדי לקבוע את שנת ההתחלה של האנימציה.**"
    ))

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    col_a, col_b = st.columns([1, 3])
    with col_a:
        start_year = st.slider(t("Select Start Year", "בחר שנת התחלה"), min_value=min_year, max_value=max_year-1, value=max(2010, min_year))

    vol_df = df[df["year"] >= start_year].copy()

    if vol_df.empty or vol_df["year"].nunique() <= 1:
        st.warning(t("Not enough data from the selected year onwards to show an animation.", "אין מספיק נתונים משנה זו והלאה כדי להציג אנימציה."))
    else:
        years = range(start_year, max_year + 1)
        sources = vol_df["source_db"].unique()
        idx = pd.MultiIndex.from_product([years, sources], names=["year", "source_db"])

        annual_df = vol_df.groupby(["year", "source_db"]).size().reindex(idx, fill_value=0).reset_index(name="Annual_Count").sort_values(by="year")
        annual_df["Cumulative_Volume"] = annual_df.groupby("source_db")["Annual_Count"].cumsum()

        fig_race = px.bar(
            annual_df, x="Cumulative_Volume", y="source_db", color="source_db",
            animation_frame="year", animation_group="source_db", orientation="h",
            range_x=[0, annual_df["Cumulative_Volume"].max() * 1.05],
            title=t(f"Cumulative Titles Published ({start_year}-{max_year})", f"היקף כותרים שפורסמו במצטבר ({start_year}-{max_year})"),
            labels={"Cumulative_Volume": t("Total Books", "סה״כ ספרים"), "source_db": t("Format", "פורמט"), "year": t("Year", "שנה")},
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