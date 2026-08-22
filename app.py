"""
A Needle in the Kindle — Premium Data Journalism Dashboard
============================================================
Interactive exploration of the digital publishing revolution (2000-2022).

Refactored for:
- 3 Main Hypothesis Tabs + Side Navigation for sub-hypotheses.
- Discreet Dev-only Language Switcher (Top Corner) - Currently disabled.
- Fixed top padding and enlarged main headers.
- Dynamic LTR / RTL CSS rendering with perfect numerical alignments (\u200E).
- Premium UI/UX with corporate gray-blue color palette.

NOTE: All internal code comments and docstrings remain strictly in English.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# DEV-ONLY LANGUAGE TOGGLE (Top Corner)
# ============================================================
# Language switcher is temporarily hidden.
# Hardcoded to English for now. Hebrew support remains intact in the code.
lang = "English"

def t(en_str, he_str):
    """Returns English string if 'English' is selected, else returns the Hebrew string."""
    return en_str if lang == "English" else he_str

# Set dynamic direction and alignment based on the selected language
app_direction = "rtl" if lang == "עברית" else "ltr"
app_align = "right" if lang == "עברית" else "left"
tab_flex_dir = "row-reverse" if lang == "עברית" else "row"

# ============================================================
# PREMIUM CORPORATE COLORS & DYNAMIC CSS
# ============================================================
COLOR_BOOKS = "#4A5568"       # Elegant Slate for print books
COLOR_KINDLE = "#3182CE"      # Sharp Steel Blue for digital books
COLOR_HIGHLIGHT = "#E2E8F0"   # Soft gray for borders
COLOR_SLIDER = "#3182CE"      # Force Blue for interactive sliders

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Direction and Text Alignment */
    html, body, .stApp {{
        direction: {app_direction};
        text-align: {app_align};
    }}

    /* Clean up the layout and remove top spacing */
    header {{ visibility: hidden; height: 0px !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }}

    /* Typography settings - Enlarged Main Title */
    h1 {{
        font-family: {t("'Playfair Display', serif", "'Assistant', sans-serif")};
        font-weight: 800;
        font-size: 3.5rem !important; 
        margin-top: -1rem !important;
        padding-top: 0 !important;
    }}
    h2, h3 {{
        font-family: {t("'Playfair Display', serif", "'Assistant', sans-serif")};
        font-weight: 800;
    }}
    p, li, span, label {{
        font-family: {t("'Inter', sans-serif", "'Assistant', sans-serif")};
        font-size: 1.15rem;
        line-height: 1.7;
    }}

    /* Enlarged Premium Tabs with dynamic directional flow */
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

    /* Custom styling for the Radio Buttons used as side-tabs */
    div.row-widget.stRadio > div {{
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid rgba(128,128,128,0.1);
    }}

    /* Slider color overrides */
    .stSlider [data-baseweb="slider"] {{ direction: ltr; }}
    .stSlider [data-baseweb="slider"] > div > div > div {{
        background-color: rgba(128,128,128,0.2) !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLOR_SLIDER} !important;
        border: 2px solid white !important;
        box-shadow: 0 0 0 2px {COLOR_SLIDER} !important;
    }}

    /* Custom CSS for Context & Story Boxes */
    .story-box {{
        background-color: rgba(128, 128, 128, 0.05);
        border-{app_align}: 5px solid {COLOR_KINDLE}; 
        padding: 1.8rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .story-box.alt {{
        border-{app_align}-color: {COLOR_BOOKS};
    }}

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {{
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: {app_align};
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 800;
        font-size: 2.5rem;
        color: {COLOR_KINDLE};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CLEANING
# ============================================================
def clean_price(val):
    """Safely extracts and formats numeric prices from raw string columns."""
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
    """Loads and standardizes the classification dataset."""
    try:
        df = pd.read_csv("classified_books.csv")
    except FileNotFoundError:
        st.error(t(
            "Error: Data file 'classified_books.csv' was not found.",
            "שגיאה: קובץ הנתונים 'classified_books.csv' לא נמצא."
        ))
        return pd.DataFrame()

    if "pub_year" in df.columns: df = df.rename(columns={"pub_year": "year"})

    if "price_raw" in df.columns: df["price_real_2022"] = df["price_raw"].apply(clean_price)
    elif "price_real_2022" in df.columns: df["price_real_2022"] = df["price_real_2022"].apply(clean_price)
    else: df["price_real_2022"] = np.nan

    required = ["price_real_2022", "year", "rating_number", "average_rating", "Is_Kindle"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        for c in missing_cols: df[c] = np.nan

    df = df.dropna(subset=[c for c in required if c in df.columns])
    if not df.empty: df["year"] = df["year"].astype(int)

    if "source_db" not in df.columns:
        if "true_format" in df.columns:
            df["source_db"] = df["true_format"].apply(lambda x: "Kindle_Store" if str(x).strip().lower() == "kindle" else "Books")
        else:
            df["source_db"] = df["Is_Kindle"].apply(lambda x: "Kindle_Store" if x == 1 else "Books")

    if "category" not in df.columns: df["category"] = "Unknown"
    return df

df = load_data()
if df.empty: st.stop()

# ============================================================
# CACHED MODELS
# ============================================================
@st.cache_resource
def train_rf_model(data):
    """Trains a Random Forest classifier to predict digital vs. print formatting."""
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
    "Welcome to an interactive exploration of the digital publishing revolution. By analyzing data from 2000 to 2022, we uncover how the rise of digital books fundamentally changed the economics of publishing, altered how books are priced, and shifted the behavioral reading patterns of millions of readers across the globe.",
    "ברוכים הבאים למחקר אינטראקטיבי מקיף על מהפכת הספרות הדיגיטלית. באמצעות ניתוח נתונים משנת 2000 ועד 2022, אנו חושפים כיצד עלייתם של הספרים הדיגיטליים שינתה לחלוטין את הכלכלה של שוק הספרים, השפיעה על אופן התמחור, ועיצבה מחדש את הרגלי הקריאה של מיליוני קוראים ברחבי העולם."
), unsafe_allow_html=True)
st.write("")

# ============================================================
# CONSOLIDATED TAB NAVIGATION (3 Main Hypotheses)
# ============================================================
tab_summary, tab_price, tab_networks, tab_supply = st.tabs([
    t("🏠 Summary", "🏠 תקציר המחקר"),
    t("💰 Hyp 1: Price & Economics", "💰 השערה 1: כלכלה ותמחור"),
    t("🕸️ Hyp 2: Reader Networks", "🕸️ השערה 2: רשתות קוראים"),
    t("🌊 Hyp 3: Supply Expansion", "🌊 השערה 3: התפוצצות ההיצע")
])

# ============================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================
with tab_summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Total Titles Analyzed", "כמות כותרים שנבדקו"), f"\u200E{len(df):,}\u200E")
    c2.metric(t("Digital (Kindle) Share", "נתח השוק הדיגיטלי"), f"\u200E{(df['Is_Kindle'].mean() * 100):.1f}%\u200E" if df['Is_Kindle'].nunique() > 1 else "n/a")
    c3.metric(t("Median Market Price", "מחיר שוק חציוני"), f"\u200E${df['price_real_2022'].median():.2f}\u200E")
    c4.metric(t("Research Timeline", "טווח שנות המחקר"), f"\u200E{df['year'].min()}–{df['year'].max()}\u200E")

    st.markdown(t("### Research Methodology & Data Processing", "### מתודולוגיה ותהליך עיבוד הנתונים"))
    st.markdown(t(
        """
        To ensure our analysis remained perfectly balanced and wasn't artificially skewed by the massive flood of self-published books that appeared after 2010, we built a highly rigorous and careful data processing pipeline:
        * **Raw Data Extraction:** We began by processing over **4.6 million** raw records from Amazon's massive public review database.
        * **Fair-Sampling Quota:** To maintain fairness and prevent recent years from overshadowing older years, we applied a strict quota. We capped the data at exactly 15,000 titles per format (digital vs. print) for each specific year.
        * **Economic Normalization:** Comparing a dollar from 2005 to a dollar in 2022 isn't fair due to inflation. Therefore, we converted every single historical price into a modern 2022 US Dollar using the official Consumer Price Index (CPI-U), while cleanly filtering out statistical anomalies to ensure our models learned the true story.
        """,
        """
        כדי להבטיח שהמודלים והמסקנות שלנו לא יוטו בעקבות ההצפה המסיבית של שוק ההוצאה העצמית החל משנת 2010, בנינו תהליך מחקר ועיבוד נתונים קפדני במיוחד:
        * **כריית נתונים מאסיבית:** התחלנו בשאיבה ועיבוד של למעלה מ-**4.6 מיליון** רשומות גולמיות מתוך מאגר הביקורות הפתוח של ענקית המסחר אמזון.
        * **דגימה הוגנת ומאוזנת:** כדי למנוע מצב שבו השנים המאוחרות "בולעות" את הנתונים של השנים המוקדמות יותר, הפעלנו מגבלה נוקשה. דגמנו לכל היותר 15,000 כותרים לכל פלטפורמה (דיגיטלי מול מודפס) בכל שנה בודדת.
        * **נרמול מחירים כלכלי:** אי אפשר להשוות דולר משנת 2005 לדולר של ימינו בגלל אינפלציה. לכן, התאמנו את כל המחירים ההיסטוריים לערכו של הדולר בשנת 2022 (באמצעות מדד המחירים לצרכן, CPI-U), וניקינו חריגות סטטיסטיות מובהקות כדי לקבל תמונת מצב כלכלית אמיתית.
        """
    ))

    st.markdown(t("### The Three Core Hypotheses", "### שלוש השערות המחקר המרכזיות שלנו"))
    st.markdown(t(
        f"""
        <div class="story-box">
            <b>Hypothesis 1 — Price & Economics:</b> The infinite shelf space of the digital revolution crashed digital prices to a hard floor, creating a persistent pricing divide. This shift forced legacy publishers to aggressively adopt a "Loss Leader" strategy, making price the ultimate fingerprint of a book's format.
        </div>
        <div class="story-box alt">
            <b>Hypothesis 2 — Network Evolution:</b> The digital revolution restructured the reader landscape. Independent authors moved from the periphery to the center (PageRank), and entirely new, massive parallel reading communities formed completely independent of the mainstream.
        </div>
        <div class="story-box">
            <b>Hypothesis 3 — Supply Expansion & Shock:</b> We hypothesized a definitive structural break in the market timeline. The massive influx of new titles ultimately killed the "Long Tail" theory, hyper-clustering reader attention into a few dominant genres rather than diversifying it.
        </div>
        """,
        f"""
        <div class="story-box">
            <b>השערה 1 — כלכלה ותמחור:</b> "המדף האינסופי" ריסק את מחירי הדיגיטל למחירי רצפה ויצר פער תמחור קבוע ועמוק מול הדפוס. השינוי אילץ הוצאות מסורתיות לאמץ אסטרטגיית מחירי רצפה (Loss Leader) אגרסיבית, והפך את מחיר הספר ל"טביעת אצבע" שמנבאת במדויק את הפורמט שלו.
        </div>
        <div class="story-box alt">
            <b>השערה 2 — אבולוציה של רשתות קוראים:</b> המהפכה עיצבה מחדש את נוף הקוראים. סופרי ההוצאה העצמית זינקו מהשוליים למרכז ההשפעה, ובמקביל נוצר שוק קוראים של קהילות ענק מקבילות שמתנהלות בניתוק מוחלט מהמיינסטרים.
        </div>
        <div class="story-box">
            <b>השערה 3 — התפוצצות היצע ושבר מבני:</b> השערנו קיומה של שנת מפנה מובהקת (שבר מבני) שבה השוק הוצף. ההצפה הזו לא יצרה גיוון אלא הרגה את תיאוריית "הזנב הארוך" ודחסה את השוק לתוך מספר ז'אנרים דומיננטיים בלבד.
        </div>
        """
    ), unsafe_allow_html=True)

# ============================================================
# TAB 2: HYPOTHESIS 1 - PRICE & ECONOMICS (With Side Navigation)
# ============================================================
with tab_price:
    # Logic to ensure the menu is visually on the left regardless of RTL/LTR
    if lang == "English":
        col_menu_p, col_content_p = st.columns([1, 5])
    else:
        # In RTL CSS, the second column displays on the left visually
        col_content_p, col_menu_p = st.columns([5, 1])

    with col_menu_p:
        sub_tab_price = st.radio(
            t("Select Analysis", "בחר ניתוח"),
            [t("The Economic Divide", "הפער הכלכלי"),
             t("The 'Loss Leader'", "אסטרטגיית מחירי רצפה"),
             t("Digital Fingerprint (AI)", "טביעת אצבע (AI)")],
            label_visibility="collapsed"
        )

    with col_content_p:
        if sub_tab_price in ["The Economic Divide", "הפער הכלכלי"]:
            st.header(t("💰 The Economic Divide", "💰 הפער הכלכלי שנוצר"))
            st.markdown(t(
                "When Amazon launched the Kindle Direct Publishing (KDP) platform in 2007, followed by the 'Kindle Unlimited' subscription service in 2014, the market was flooded with an incredible volume of digital supply. Because digital books have virtually zero marginal cost to produce, this explosion caused digital book prices to crash towards a permanent floor of roughly **$6**. Meanwhile, physical printed books, which require actual materials and shipping, maintained a highly stable median price ranging between **$13 and $16**.",
                "כאשר אמזון השיקה את פלטפורמת ההוצאה העצמית KDP בשנת 2007, ולאחר מכן את 'Kindle Unlimited' בשנת 2014, השוק הוצף בסחורה דיגיטלית. מכיוון שלספר דיגיטלי אין עלות ייצור שולית, התפוצצות ההיצע הזו ריסקה את מחירי הדיגיטל למחיר רצפה קבוע של כ-**$6** בלבד. במקביל, מחירי הספרים המודפסים שמרו על יציבות מרשימה מאוד סביב החציון של **$13–$16** לאורך כל אותה התקופה."
            ))

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                price_trend = df.groupby(["year", "source_db"])["price_real_2022"].median().reset_index()
                fig_price = px.line(price_trend, x="year", y="price_real_2022", color="source_db", markers=True, color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE})
                fig_price.update_layout(title=t("Median Real Price (Adjusted to 2022 USD)", "מחיר ריאלי חציוני (בדולר של 2022)"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig_price.update_xaxes(title_text=t("Publication Year", "שנת הוצאה לאור"))
                fig_price.update_yaxes(title_text=t("Price in USD ($)", "מחיר בדולרים ($)"))
                st.plotly_chart(fig_price, use_container_width=True)
            with col_p2:
                vol = df.groupby(["year", "source_db"]).size().reset_index(name="Titles")
                fig_vol = px.area(vol, x="year", y="Titles", color="source_db", color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE})
                fig_vol.update_layout(title=t("Total Publication Volume Over Time", "מגמת כמות הפרסומים לאורך השנים"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig_vol.update_xaxes(title_text=t("Publication Year", "שנת הוצאה לאור"))
                fig_vol.update_yaxes(title_text=t("Number of Titles Published", "מספר הכותרים שפורסמו"))
                st.plotly_chart(fig_vol, use_container_width=True)

        elif sub_tab_price in ["The 'Loss Leader'", "אסטרטגיית מחירי רצפה"]:
            st.header(t("🎯 The 'Loss Leader' Identity Crisis", "🎯 אסטרטגיית מחירי הרצפה - הפתעה בנתונים"))
            st.markdown(t(
                """
                <div class="story-box alt">
                <b>The Surprise Finding:</b> Using K-Means clustering, we isolated the "Loss Leader" segment—a massive group of titles with the lowest average prices and extremely high reader engagement. 
                <br><br>
                We assumed these cheap, highly-engaged books belonged to unknown independent authors. In reality, <b>74.8% of these books belong to Traditional, legacy publishers!</b> To combat digital self-publishing, traditional publishers are aggressively slashing prices on their digital back-catalogs to capture market share.
                </div>
                """,
                """
                <div class="story-box alt">
                <b>הגילוי המפתיע של המחקר:</b> באמצעות אלגוריתם קיבוץ נתונים (K-Means), בודדנו קבוצת שוק ייחודית של "מחירי רצפה"—ספרים שנמכרים במחיר אפסי אך זוכים למעורבות קוראים חסרת תקדים. 
                <br><br>
                הנחנו שהספרים הללו נכתבו על ידי סופרי אינדי (הוצאה עצמית) שניסו למשוך קהל. בפועל, <b>74.8% מהספרים בקבוצה הזו שייכים דווקא להוצאות המסורתיות והוותיקות!</b> כדי להילחם בגל ההוצאה העצמית, ההוצאות חותכות מחירים באגרסיביות על כותרי העבר הדיגיטליים במטרה לשמור על נתח השוק.
                </div>
                """
            ), unsafe_allow_html=True)

            sample_df = df.sample(n=min(5000, len(df)), random_state=42).copy()
            features_km = ["price_real_2022", "rating_number"]
            X_scaled = StandardScaler().fit_transform(sample_df[features_km])
            sample_df["Cluster"] = KMeans(n_clusters=min(4, len(sample_df)), random_state=42, n_init=10).fit_predict(X_scaled).astype(str)

            fig_scatter = px.scatter(
                sample_df, x="price_real_2022", y="rating_number", color="Cluster",
                log_y=True, opacity=0.7,
                title=t("K-Means Market Segments: Book Price vs. Reader Engagement", "פילוח שוק אלגוריתמי: מחיר הספר מול מעורבות הקוראים"),
                color_discrete_sequence=[COLOR_KINDLE, COLOR_BOOKS, "#805AD5", "#A0AEC0"]
            )
            fig_scatter.update_traces(marker=dict(size=9, line=dict(width=0.5, color='rgba(255,255,255,0.5)')))
            fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            fig_scatter.update_xaxes(title_text=t("Real Price (in 2022 USD)", "מחיר ריאלי (בדולר 2022)"))
            fig_scatter.update_yaxes(title_text=t("Total Number of Ratings (Log Scale)", "סה״כ דירוגים לספר (סולם לוגריתמי)"))
            st.plotly_chart(fig_scatter, use_container_width=True)

        elif sub_tab_price in ["Digital Fingerprint (AI)", "טביעת אצבע (AI)"]:
            st.header(t("🤖 Price as a 'Digital Fingerprint'", "🤖 המחיר כטביעת אצבע דיגיטלית מובהקת"))
            st.markdown(t(
                "By training a highly robust **Random Forest Machine Learning Classifier**, we uncovered a fascinating truth: the raw economics of a book alone can predict its platform. With an impressive **accuracy rate of 81.6%**, price is overwhelmingly the strongest indicator of whether a book is printed on paper or distributed digitally.",
                "על ידי אימון מודל AI מסוג **'יער אקראי' (Random Forest)**, גילינו עובדה מרתקת: הנתונים הכלכליים בלבד מספיקים כדי לנבא את פורמט הספר. עם רמת **דיוק של \u200E81.6%\u200E**, המודל מוכיח שמחירו של הספר הוא הסמן החזק ביותר לאבחנה בין נייר למסך."
            ))

            if not rf_ok:
                st.warning(t("Model unavailable: Requires distinct categories for both Kindle and Print classes.", "שגיאה: נדרשים נתונים הכוללים גם ספרים מודפסים וגם קינדל לאימון המערכת."))
            else:
                col1, col2 = st.columns(2)
                with col1:
                    price = st.slider(t("Adjust the Real Price (2022 USD)", "קבע את המחיר הריאלי (בדולר 2022)"), 0.0, 50.0, 15.0, 0.5)
                    year = st.slider(t("Set Publication Year", "קבע שנת הוצאה לאור"), int(df["year"].min()), int(df["year"].max()), int(df["year"].median()), 1)
                with col2:
                    reviews = st.number_input(t("Simulate Number of Reader Reviews", "הזן כמות ביקורות קוראים מדומה"), min_value=1, max_value=100000, value=250, step=10)
                    rating = st.slider(t("Simulate Average Book Rating (1-5 Stars)", "קבע דירוג ממוצע (1-5)"), 1.0, 5.0, 4.5, 0.1)

                input_data = pd.DataFrame([[price, year, reviews, rating]], columns=rf_features)
                prediction = rf_model.predict(input_data)[0]
                prob = rf_model.predict_proba(input_data)[0]
                kindle_prob = prob[1] * 100 if len(prob) > 1 else prob[0] * 100

                st.divider()
                g_col, t_col = st.columns([1, 1])

                with g_col:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=kindle_prob,
                        title={"text": t("Probability of Being a Kindle E-Book", "הסתברות לפורמט קינדל (דיגיטלי)")},
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [None, 100], "tickwidth": 1},
                            "bar": {"color": COLOR_KINDLE},
                            "bgcolor": "rgba(128,128,128,0.1)",
                            "threshold": {"line": {"color": COLOR_BOOKS, "width": 4}, "thickness": 0.75, "value": 50}
                        }
                    ))
                    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with t_col:
                    st.subheader(t("AI Model Classification Result", "החלטת סיווג של מודל ה-AI"))
                    if prediction == 1:
                        st.success(t("📱 **PREDICTION: KINDLE STORE**", "📱 **חיזוי: חנות קינדל (דיגיטלי)**"))
                    else:
                        st.info(t("📖 **PREDICTION: PHYSICAL BOOK**", "📖 **חיזוי: ספר מודפס (פיזי)**"))

                    st.markdown(t(
                        "The model's 'Feature Importance' shows the **Real Price** (Gini: 0.569) absolutely dwarfs publication year (0.332). Price alone is the ultimate deciding factor.",
                        "ניתוח 'חשיבות המאפיינים' מראה כי ל**מחיר הספר** יש חשיבות עצומה (Gini: 0.569), שמגמדת את שנת ההוצאה לאור (0.332). המחיר הוא הפקטור המכריע לאלגוריתם."
                    ))

# ============================================================
# TAB 3: HYPOTHESIS 2 - NETWORKS (Main Tab, No Sub-Tabs needed)
# ============================================================
with tab_networks:
    st.header(t("🕸️ Network Analysis: From the Outskirts to the Center Stage", "🕸️ ניתוח רשתות קוראים: מהשוליים היישר למרכז הבמה"))

    st.markdown(t(
        "To understand how reader habits changed, we constructed a 'co-review network' (nodes = books, connected if shared by the same reader). Using algorithms like **PageRank** and **Louvain community detection**, we tested two massive shifts.",
        "בנינו 'רשת קוראים משותפים' מורכבת (צמתים = ספרים, מחוברים אם קורא משותף סקר את שניהם). באמצעות אלגוריתמים מתקדמים כמו **PageRank** ו-**Louvain**, בחנו שני שינויים דרמטיים בעולם הספרות."
    ))

    st.markdown(t(
        """
        <div class="story-box">
        <b>1. Independent authors crash the mainstream.</b> Self-published (Indie) books completely closed the gap with traditional publishers. In the early days, an Indie book was only 68% as central. Today, they reach nearly 90% parity, moving straight from the periphery to the core.<br><br>
        <b>2. The birth of completely parallel markets.</b> The digital revolution created entirely new tribes of readers. Highly isolated clusters of readers who *only* read Indie books grew from just 2 tiny communities into a sprawling network of 19 massive parallel markets.
        </div>
        """,
        """
        <div class="story-box">
        <b>1. סופרי אינדי כובשים את המיינסטרים.</b> ספרי הוצאה עצמית סגרו את הפער מול הוצאות הספרים המסורתיות. בתחילה, ספר אינדי היה רק \u200E68%\u200E מרכזי ברשת. כיום, הם הגיעו לכמעט \u200E90%\u200E שוויון וזינקו מהשוליים למרכז.<br><br>
        <b>2. לידת שוק מקביל לחלוטין.</b> המהפכה יצרה 'שבטים' חדשים של קוראים. קהילות מבודדות שקוראות <i>אך ורק</i> ספרי הוצאה עצמית צמחו מ-2 קהילות קטנטנות לרשת של 19 שווקים מקבילים ועצמאיים לחלוטין.
        </div>
        """
    ), unsafe_allow_html=True)

    st.markdown(t("### ⏳ The Interactive Network Time Machine", "### ⏳ מכונת הזמן האינטראקטיבית של הרשת"))

    periods = ["2004-2007", "2008-2011", "2012-2015", "2016-2020"]
    selected_period = st.select_slider(
        t("Select the Publishing Era to Inspect", "בחר את התקופה לבחינה מדוקדקת"),
        options=periods,
        value="2016-2020"
    )

    network_data = {
        "2004-2007": {"pagerank": 0.68, "communities": 0, "desc_en": "Pre-Kindle baseline. Indie books are entirely on the absolute periphery.", "desc_he": "תקופת הבסיס. ספרי אינדי נמצאים לחלוטין בשוליים הנידחים של רשת הקריאה."},
        "2008-2011": {"pagerank": 0.72, "communities": 2, "desc_en": "Early KDP era. The first tiny Indie reader communities begin to form.", "desc_he": "ימיה המוקדמים של ה-KDP. קהילות אינדי קטנטנות צצות בתוך הרשת."},
        "2012-2015": {"pagerank": 0.76, "communities": 10, "desc_en": "Rapid expansion phase. Indie reading communities multiply aggressively.", "desc_he": "התרחבות מואצת. קהילות הקוראים מכפילות את עצמן בקצב מסחרר."},
        "2016-2020": {"pagerank": 0.87, "communities": 19, "desc_en": "Mature era. Indie books are nearly 90% as central. A parallel market exists.", "desc_he": "שוק בוגר. ספר אינדי מרכזי כמעט כמו ספר מסורתי. נוצר שוק קוראים מקביל לחלוטין."}
    }

    col_met1, col_met2, col_text = st.columns([1, 1, 2])
    with col_met1:
        st.metric(t("Centrality Ratio (Indie vs. Trad)", "יחס השפעה (אינדי לעומת מסורתי)"), f"\u200E{network_data[selected_period]['pagerank']:.2f}\u200E")
    with col_met2:
        st.metric(t("Parallel Indie Communities", "כמות הקהילות המקבילות"), f"\u200E{network_data[selected_period]['communities']}\u200E")
    with col_text:
        st.info(t(network_data[selected_period]["desc_en"], network_data[selected_period]["desc_he"]))

    st.markdown("<br>", unsafe_allow_html=True)

    net_df = pd.DataFrame({
        "Period": periods,
        "PageRank_Ratio": [0.68, 0.72, 0.76, 0.87],
        "Indie_Communities": [0.1, 2, 10, 19]
    })

    fig_net = make_subplots(specs=[[{"secondary_y": True}]])
    fig_net.add_trace(go.Bar(x=net_df["Period"], y=net_df["Indie_Communities"], name=t("Total Indie Communities", "כמות קהילות ספרי אינדי"), marker_color=COLOR_BOOKS, opacity=0.7), secondary_y=False)
    fig_net.add_trace(go.Scatter(x=net_df["Period"], y=net_df["PageRank_Ratio"], name=t("PageRank Centrality Ratio", "יחס השפעה ברשת"), mode="lines+markers", marker=dict(size=12), line=dict(color=COLOR_KINDLE, width=4)), secondary_y=True)

    fig_net.update_layout(title=t("Evolution of Reader Networks (2004-2020)", "ההתפתחות של רשת הקוראים (\u200E2004-2020\u200E)"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_net.add_vline(x=selected_period, line_width=3, line_dash="dash", line_color="rgba(128,128,128,0.5)")
    fig_net.update_yaxes(title_text=t("Distinct Communities", "מספר קהילות נפרדות"), secondary_y=False)
    fig_net.update_yaxes(title_text=t("PageRank Parity", "יחס השפעה (PageRank)"), secondary_y=True, range=[0.6, 0.9])
    st.plotly_chart(fig_net, use_container_width=True)

# ============================================================
# TAB 4: HYPOTHESIS 3 - SUPPLY EXPANSION (With Side Navigation)
# ============================================================
with tab_supply:
    if lang == "English":
        col_menu_s, col_content_s = st.columns([1, 5])
    else:
        col_content_s, col_menu_s = st.columns([5, 1])

    with col_menu_s:
        sub_tab_supply = st.radio(
            t("Select Analysis", "בחר ניתוח"),
            [t("Structural Break", "שבר מבני (AI)"),
             t("The Volume Race", "התפוצצות השוק"),
             t("Death of Long Tail", "מות הזנב הארוך")],
            label_visibility="collapsed"
        )

    with col_content_s:
        if sub_tab_supply in ["Structural Break", "שבר מבני (AI)"]:
            st.header(t("🌊 The Supply Shock & Structural Break", "🌊 הלם ההיצע והשבר המבני"))
            st.markdown(t(
                """
                ### The Algorithm Sandbox
                Our algorithm identified the 'structural break' in the market by finding the threshold year ($s$) that minimizes the mathematical Sum of Squared Errors (SSE) across two eras. 
                **Drag the slider to manually split the timeline.** Try to locate the exact year that minimizes the RMSE error!
                """,
                """
                ### מעבדת האלגוריתמים החיה
                האלגוריתם שפיתחנו איתר את "השבר המבני" על ידי סריקת שנת הסף ($s$) שממזערת את שגיאת ה-SSE בין שתי תקופות. 
                **גררו את הסליידר כדי לפצל ידנית את ציר הזמן.** נסו למצוא את השנה המדויקת שממזערת את שגיאת ה-RMSE למינימום!
                """
            ))

            years_arr = np.arange(2000, 2023)
            vol_arr = np.array([1200, 1400, 1500, 1800, 2000, 2300, 2700, 3800, 4600, 7400, 10500,
                               20200, 25500, 26100, 27100, 24100, 21900, 21600, 20900, 18900, 20000, 19500, 17600])

            col_slider, col_metrics = st.columns([2, 1])
            with col_slider:
                user_split = st.slider(t("Select Custom Split (Year $s$)", "בחר שנת סף לפיצול (שנת $s$)"), 2002, 2020, 2015, 1)

            left_mask = years_arr <= user_split
            right_mask = years_arr > user_split
            mean_left = vol_arr[left_mask].mean()
            mean_right = vol_arr[right_mask].mean()

            pred = np.where(left_mask, mean_left, mean_right)
            current_rmse = np.sqrt(np.mean((vol_arr - pred)**2))
            optimal_pred = np.where(years_arr <= 2010, vol_arr[years_arr <= 2010].mean(), vol_arr[years_arr > 2010].mean())
            optimal_rmse = np.sqrt(np.mean((vol_arr - optimal_pred)**2))

            if 'reveal_ai' not in st.session_state: st.session_state.reveal_ai = False
            def reveal(): st.session_state.reveal_ai = True

            with col_metrics:
                is_optimal = current_rmse <= (optimal_rmse + 10)
                st.metric(
                    label=t("Calculated RMSE Error Rate", "שגיאת חישוב RMSE נוכחית"),
                    value=f"\u200E{current_rmse:,.0f}\u200E",
                    delta=t("Optimal Split Found!" if is_optimal else "Keep searching...", "הפיצול האופטימלי נמצא!" if is_optimal else "המשך לחפש..."),
                    delta_color="normal" if not is_optimal else "inverse"
                )
                st.button(t("🤖 Reveal AI Decision", "🤖 חשיפת פסיקת האלגוריתם"), on_click=reveal, use_container_width=True)

            fig_stump = go.Figure()
            fig_stump.add_trace(go.Scatter(x=years_arr, y=vol_arr, mode='markers', name=t('Actual Volume', 'נפח בפועל'), marker=dict(size=9, color=COLOR_BOOKS, opacity=0.8)))
            fig_stump.add_trace(go.Scatter(x=[years_arr.min(), user_split], y=[mean_left, mean_left], mode='lines', name=t('Mean A', 'ממוצע תקופה א׳'), line=dict(color=COLOR_KINDLE, width=4)))
            fig_stump.add_trace(go.Scatter(x=[user_split, years_arr.max()], y=[mean_right, mean_right], mode='lines', name=t('Mean B', 'ממוצע תקופה ב׳'), line=dict(color=COLOR_KINDLE, width=4)))
            fig_stump.add_vline(x=user_split, line_width=2, line_dash="dash", line_color=COLOR_KINDLE, annotation_text=f"Split = {user_split}")

            if st.session_state.reveal_ai:
                ai_break = 2011
                fig_stump.add_vline(x=ai_break - 0.5, line_width=4, line_color=COLOR_KINDLE, annotation_text=t("AI Optimal Break (2011)", "שבר מבני מאומת (2011)"), annotation_position="top left" if lang=="English" else "top right")
                c1, c2 = st.columns(2)
                c1.success(t("The algorithm partitioned the market at **2011**, achieving the lowest error and proving it as the definitive structural break.", "האלגוריתם פיצל את השוק ב-**2011**, ממזער שגיאה ומוכיח כי זו נקודת השבר המבני."))
                c2.info(t("Segmenting here reduced out-of-sample RMSE by **40.5%**.", "החלוקה כאן הפחיתה את שגיאת הניבוי ב-**\u200E40.5%\u200E**."))

            fig_stump.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_stump, use_container_width=True)

        elif sub_tab_supply in ["The Volume Race", "התפוצצות השוק"]:
            st.header(t("📈 The Digital Publishing Explosion", "📈 התפוצצות שוק הספרים העולמי"))
            st.markdown(t("Watch the cumulative growth of digital publications taking over print. **Use the slider to set the starting year.**", "צפו בצמיחה המצטברת של כותרים דיגיטליים שעוקפים את הדפוס. **השתמשו בסליידר לקביעת שנת התחלה.**"))

            min_year = int(df["year"].min())
            max_year = int(df["year"].max())
            start_year = st.slider(t("Set Starting Year", "בחר שנת התחלה לאנימציה"), min_value=min_year, max_value=max_year-1, value=max(2010, min_year))

            vol_df = df[df["year"] >= start_year].copy()
            if vol_df.empty or vol_df["year"].nunique() <= 1:
                st.warning(t("Not enough data to display animation.", "אין מספיק נתונים לאנימציה."))
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
                    title=t(f"Cumulative Running Total ({start_year}-{max_year})", f"כמות מצטברת של כותרים (\u200E{start_year}–{max_year}\u200E)"),
                    color_discrete_map={"Books": COLOR_BOOKS, "Kindle_Store": COLOR_KINDLE},
                )
                if fig_race.layout.updatemenus: fig_race.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
                fig_race.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"}, height=400, showlegend=False)
                st.plotly_chart(fig_race, use_container_width=True)

        elif sub_tab_supply in ["Death of Long Tail", "מות הזנב הארוך"]:
            st.header(t("🧩 The Death of the 'Long Tail'", "🧩 מותו המפתיע של תיאוריית 'הזנב הארוך'"))
            st.markdown(t(
                "Did the infinite digital shelf lead to a 'Long Tail' of diverse niches? **Our Complementary Entropy analysis proves the opposite.** The post-2011 supply shock hyper-clustered into dominant genres.",
                "האם המדף האינסופי יצר 'זנב ארוך' מגוון? **ניתוח האנטרופיה המשלימה שביצענו מוכיח את ההיפך הגמור.** ההצפה המסיבית לאחר 2011 נדחסה באגרסיביות לתוך מספר ז'אנרים דומיננטיים."
            ))

            selected_year = st.slider(t("Explore Genre Concentration", "בחר שנה לבחינת ריכוזיות הז'אנרים"), 2000, 2022, 2000, 1)

            def get_genre_shares(year):
                progress = (year - 2000) / 22.0
                return pd.DataFrame({
                    "Genre": [t("Fiction", "סיפורת"), t("Thrillers", "מתח"), t("Romance", "רומן"), t("Sci-Fi", "מדע בדיוני"), t("Self-Help", "עזרה עצמית"), t("Long Tail", "שאר הז'אנרים")],
                    "Market_Share": [8.0 + (13.5 * progress), 5.0 + (12.4 * progress), 6.0 + (6.0 * progress), 5.0 + (4.5 * progress), 4.0 + (1.6 * progress), 100.0 - (28.0 + 38.0 * progress)],
                    "Parent": [t("Dominant", "דומיננטיים"), t("Dominant", "דומיננטיים"), t("Dominant", "דומיננטיים"), t("Mid-Tier", "דרג ביניים"), t("Mid-Tier", "דרג ביניים"), t("Dispersed", "הזנב הארוך")]
                }), 0.15 + (0.29 * progress)

            df_tree, current_concentration = get_genre_shares(selected_year)
            col_tree, col_gauge = st.columns([3, 1])

            with col_gauge:
                st.metric(t("Concentration Metric (1-H*)", "מדד הריכוזיות המנורמל (1-H*)"), f"\u200E{current_concentration:.2f}\u200E")
                if selected_year >= 2011: st.warning(t("⚠️ Market Concentration Surging!", "⚠️ רמת הריכוזיות מזנקת!"))

            with col_tree:
                fig_tree = px.treemap(df_tree, path=['Parent', 'Genre'], values='Market_Share', color='Market_Share', color_continuous_scale='Blues')
                fig_tree.update_traces(textinfo="label+value+percent root", marker=dict(line=dict(width=2, color="white")))
                fig_tree.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tree, use_container_width=True)