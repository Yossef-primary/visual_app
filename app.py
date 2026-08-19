"""
Interactive Dashboard for 'A Needle in the Kindle'
==================================================
This Streamlit application provides an interactive exploration of the digital
publishing revolution. It features a live Random Forest predictor,
interactive K-Means clustering, and a dynamic bar chart race.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration for a premium layout
st.set_page_config(page_title="A Needle in the Kindle", layout="wide", page_icon="📚")

# Upgrade 1: Custom CSS for a premium data journalism aesthetic
st.markdown("""
    <style>
    /* Main background and typography */
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Customizing sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    
    /* Styling sliders and progress elements */
    .stSlider > div > div > div > div { background-color: #e67e22 !important; }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] { color: #4b0082; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Loads and cleans the dataset. Caches the result to prevent reloading
    during active user sessions.
    """
    try:
        df = pd.read_csv("classified_books.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("classified_books.csv")
        except FileNotFoundError:
            st.error("Data file not found. Please ensure 'classified_books.csv' is in the directory.")
            st.stop()

    # Basic data cleaning and null dropping
    df = df.dropna(subset=['price_real_2022', 'year', 'rating_number', 'average_rating'])
    return df


df = load_data()


@st.cache_resource
def train_rf_model(data):
    """
    Trains the Random Forest classifier. Caches the trained model globally
    so it does not re-train upon UI interactions.
    """
    features = ['price_real_2022', 'year', 'rating_number', 'average_rating']
    X = data[features]
    y = data['Is_Kindle']

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, features


rf_model, rf_features = train_rf_model(df)

# Dashboard Title
st.title("📚 A Needle in the Kindle: Market Dashboard")
st.markdown("Explore the economic disruption of the digital publishing revolution through machine learning.")
st.divider()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. The ML Predictor", "2. Cluster Explorer", "3. Volume Over Time"])

# ---------------------------------------------------------
# PAGE 1: The ML Predictor
# ---------------------------------------------------------
if page == "1. The ML Predictor":
    st.header("🎯 Interactive Market Predictor")
    st.markdown("Adjust the variables below to see how the Random Forest algorithm classifies the publication.")

    col1, col2 = st.columns(2)

    with col1:
        price = st.slider("Real Price (2022 USD)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        year = st.slider("Publication Year", min_value=2000, max_value=2022, value=2015, step=1)

    with col2:
        reviews = st.number_input("Number of Ratings", min_value=1, max_value=100000, value=250, step=10)
        rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

    # Model inference
    input_data = pd.DataFrame([[price, year, reviews, rating]], columns=rf_features)
    prediction = rf_model.predict(input_data)[0]
    prob = rf_model.predict_proba(input_data)[0]

    st.divider()

    # Layout for the Gauge Chart and textual outcome
    gauge_col, text_col = st.columns([1, 1])

    with gauge_col:
        # Upgrade 3: Plotly Gauge Chart replacing the basic st.progress
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of Kindle Release (%)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#e67e22"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ecf0f1",
                'steps': [
                    {'range': [0, 50], 'color': '#f8f9fa'},
                    {'range': [50, 100], 'color': '#e8f4f8'}
                ],
                'threshold': {
                    'line': {'color': "#2c3e50", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with text_col:
        st.subheader("Model Classification")
        if prediction == 1:
            st.success("📱 **CLASSIFIED AS: KINDLE STORE**")
            st.markdown("The algorithm determines this combination of price, recency, and engagement is highly characteristic of a digital release.")
        else:
            st.info("📖 **CLASSIFIED AS: PHYSICAL BOOK**")
            st.markdown("The algorithm determines this profile aligns more closely with traditional print publishing economics.")

        # Upgrade 4: Data Explainer Tooltip
        with st.expander("How does the Random Forest decide?"):
            st.write("""
            The model uses a collection of **Decision Trees** that look at historical thresholds. 
            For example, extremely low prices (under $4.99) combined with a high volume of reviews 
            frequently trigger the 'Kindle' classification due to independent authors utilizing low-cost volume strategies.
            """)

# ---------------------------------------------------------
# PAGE 2: Cluster Explorer
# ---------------------------------------------------------
elif page == "2. Cluster Explorer":
    st.header("🧩 Interactive Market Segments")
    st.markdown("Hover over the scatter points to discover distinct publisher strategies based on K-Means clustering.")

    # Train KMeans on a sample for performance
    sample_df = df.sample(n=min(5000, len(df)), random_state=42).copy()
    features_km = ['price_real_2022', 'rating_number']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df[features_km])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    sample_df['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)

    # Plotly Scatter Plot
    fig_scatter = px.scatter(
        sample_df,
        x='price_real_2022',
        y='rating_number',
        color='Cluster',
        hover_data=['year', 'Is_Kindle'],
        log_y=True,
        title="Price vs. Engagement (Log Scale)",
        labels={'price_real_2022': 'Real Price (2022 USD)', 'rating_number': 'Total Ratings'},
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Upgrade 4: Explainer section for business terminology
    st.info("💡 **Understanding the Segments (Clusters):**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**The Loss Leader:** High engagement, near-zero price. Authors give away the first book in a series for free to hook readers.")
        st.markdown("**Premium Print:** High price, moderate engagement. Typically academic textbooks or niche non-fiction.")
    with col_b:
        st.markdown("**The Sweet Spot:** Mid-tier pricing ($9-$14) with consistent reviews. Standard commercial fiction.")
        st.markdown("**The Long Tail:** Low price, low engagement. Independent books struggling to find an audience.")

# ---------------------------------------------------------
# PAGE 3: Volume Over Time
# ---------------------------------------------------------
elif page == "3. Volume Over Time":
    st.header("📈 The Digital Publishing Explosion")

    # Create a complete grid of years and sources to ensure smooth animation
    years = range(df['year'].min(), df['year'].max() + 1)
    sources = df['source_db'].unique()
    idx = pd.MultiIndex.from_product([years, sources], names=['year', 'source_db'])

    # Aggregate annual volume and reindex to fill missing years with 0
    annual_df = df.groupby(['year', 'source_db']).size().reindex(idx, fill_value=0).reset_index(name='Annual_Count')
    annual_df = annual_df.sort_values(by=['year'])

    # Calculate cumulative volume for the race effect
    annual_df['Cumulative_Volume'] = annual_df.groupby('source_db')['Annual_Count'].cumsum()

    # Upgrade 2: Animated Bar Chart Race using Plotly
    fig_race = px.bar(
        annual_df,
        x="Cumulative_Volume",
        y="source_db",
        color="source_db",
        animation_frame="year",
        animation_group="source_db",
        orientation='h',
        range_x=[0, annual_df['Cumulative_Volume'].max() * 1.05],
        title="Cumulative Titles Published (2000-2022)",
        labels={'Cumulative_Volume': 'Total Books Published', 'source_db': 'Format', 'year': 'Year'},
        color_discrete_map={'Books': '#4b0082', 'Kindle_Store': '#e67e22'}
    )

    # Speed up the animation and clean up the layout
    fig_race.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
    fig_race.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )

    st.plotly_chart(fig_race, use_container_width=True)

    st.caption("Press 'Play' on the timeline axis above to watch the Kindle store rapidly overtake traditional books.")