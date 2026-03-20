import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* Target the specific container for the main title */
    [data-testid="stHeading"] h1 {
        text-align: center !important;
        color: #7b8eab !important;
    }

    /* Target the custom subtitle class */
    .subtitle {
        text-align: center !important;
        color: #a0a8b9 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Exploratory Data Analysis")
st.markdown('<p class="subtitle">This analysis examines the key trends, outliers, and correlations within the property dataset.</p>', unsafe_allow_html=True)
# Set seaborn style for all plots
sns.set_theme(style="whitegrid")

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('house_dataset.csv')
        # Fill missing values for robustness in plots
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].median(), inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'house_dataset.csv' not found. Please ensure it's in the main folder.")
        return None

df = load_data()

if df is not None:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- 1. Correlation Heatmap ---
    st.subheader("Correlation Heatmap of Numerical Features")
    st.markdown("This heatmap shows the relationship between numerical features. A darker color indicates a stronger correlation.")
    
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Correlation Heatmap", fontsize=16)
    st.pyplot(fig_corr)
    st.markdown("---")

    # --- 2. Histograms for Numerical Features ---
    st.subheader("Distribution of Numerical Features")
    st.markdown("Select a feature from the dropdown to see how its values are distributed.")
    
    selected_hist_col = st.selectbox("Select a Numeric Feature for Histogram:", numeric_cols)
    
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_hist_col], kde=True, bins=30, ax=ax_hist)
    ax_hist.set_title(f'Distribution of {selected_hist_col}', fontsize=15)
    st.pyplot(fig_hist)
    st.markdown("---")

    # --- 3. Boxplots to Show Outliers ---
    st.subheader("Detecting Outliers in Key Features")
    st.markdown("Boxplots help us easily identify extreme (very high or low) values, also known as **outliers**.")

    boxplot_features = ['Price_INR', 'SuperBuiltUpArea_sqft', 'BuiltUpArea_sqft', 'CarpetArea_sqft', 'YearBuilt', 'Bathrooms', 'Balconies']
    selected_boxplot_col = st.selectbox("Select a Feature to see its Boxplot for Outliers:", boxplot_features)

    fig_box, ax_box = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df[selected_boxplot_col], ax=ax_box)
    ax_box.set_title(f'Boxplot of {selected_boxplot_col} to find Outliers', fontsize=16)
    st.pyplot(fig_box)
    st.markdown("---")

    # --- 4. Scatter Plots vs. Price ---
    st.subheader("Relationship of Features with Price")
    st.markdown("See how different features affect the property price.")

    scatter_options = [col for col in numeric_cols if col != 'Price_INR']
    selected_scatter_col = st.selectbox("Select a Feature to plot against Price:", scatter_options)

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df[selected_scatter_col], y=df['Price_INR'], ax=ax_scatter)
    ax_scatter.set_title(f'{selected_scatter_col} vs. Price_INR')
    st.pyplot(fig_scatter)

else:
    st.warning("Data could not be loaded. 'house_dataset.csv' is required for analysis.")