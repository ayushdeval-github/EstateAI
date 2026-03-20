import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Page configuration
st.set_page_config(page_title="Price Trend and Future Forecast", page_icon="📈", layout="wide")

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
st.title("Price Trend and Future Forecast")

st.markdown('<p class="subtitle">This page replicates your custom trend analysis and ARIMA forecast.</p>', unsafe_allow_html=True)
# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('house_dataset.csv')
        df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())
        return df
    except FileNotFoundError:
        st.error("Error: 'house_dataset.csv' not found.")
        return None

df_original = load_data()

if df_original is not None:
    city = st.selectbox("Select a City to Analyze:", df_original['City'].unique())
    df = df_original[df_original['City'] == city]

    # --- Section 1: Your Custom Trend Analysis Plot ---
    st.subheader(f"Historical Price Trends for {city}")
    
    yearly_df = df.groupby('YearBuilt')['Price_INR'].mean().reset_index()
    yearly_df.set_index('YearBuilt', inplace=True)

    if not yearly_df.empty:
        # Create the plot figure from your code
        fig_trends, ax_trends = plt.subplots(figsize=(15, 8))
        
        ax_trends.scatter(df['YearBuilt'], df['Price_INR'], color='gray', alpha=0.3, s=20, label='Raw Data')
        ax_trends.plot(yearly_df.index, yearly_df['Price_INR'], label='Yearly Avg Price', color='blue', linewidth=2)
        
        # Moving Averages
        yearly_df['SMA_7'] = yearly_df['Price_INR'].rolling(window=7, min_periods=1).mean()
        yearly_df['SMA_30'] = yearly_df['Price_INR'].rolling(window=30, min_periods=1).mean()
        ax_trends.plot(yearly_df.index, yearly_df['SMA_7'], label='7-year SMA', color='orange', linewidth=2)
        ax_trends.plot(yearly_df.index, yearly_df['SMA_30'], label='30-year SMA', color='green', linewidth=2)

        # Exponential Moving Averages
        yearly_df['EMA_7'] = yearly_df['Price_INR'].ewm(span=7, adjust=False).mean()
        yearly_df['EMA_30'] = yearly_df['Price_INR'].ewm(span=30, adjust=False).mean()
        ax_trends.plot(yearly_df.index, yearly_df['EMA_7'], label='7-year EMA', color='red', linestyle='--', linewidth=2)
        ax_trends.plot(yearly_df.index, yearly_df['EMA_30'], label='30-year EMA', color='purple', linestyle='--', linewidth=2)

        # Regression Trend Line
        x = yearly_df.index.values
        y = yearly_df['Price_INR'].values
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.polyval(coeffs, x)
        ax_trends.plot(x, trend_line, label='Regression Trend Line', color='black', linestyle='-', linewidth=2)

        # Formatting
        ax_trends.legend()
        ax_trends.set_title(f"House Price Trend in {city} (SMA, EMA & Regression)", fontsize=16)
        ax_trends.set_xlabel("Year Built")
        ax_trends.set_ylabel("Price (INR)")
        ax_trends.grid(True, linestyle="--", alpha=0.6)
        
        st.pyplot(fig_trends)
    else:
        st.warning("Not enough data to display trends for this city.")

    st.markdown("---")

    # --- Section 2: Your Custom ARIMA Forecast ---
    st.subheader(f"5-Year Price Forecast for {city} (ARIMA)")
    
    if st.button("Generate Detailed Forecast"):
        ts_data = yearly_df['Price_INR']
        
        if len(ts_data) < 15:
            st.error("Not enough data to forecast. At least 15 years of data is required.")
        else:
            with st.spinner(f"Running full ARIMA analysis for {city}..."):
                
                # Fit the ARIMA model first to use for diagnostics
                ts_data.index = pd.to_datetime(ts_data.index, format='%Y')
                ts_data = ts_data.asfreq('AS-JAN')
                model = ARIMA(ts_data, order=(0, 1, 0), trend='t').fit()

                with st.expander("Show Detailed ARIMA Analysis"):
                    st.markdown("#### Stationarity Check (ADF Test)")
                    # Run test on original series
                    st.write("**Original Series:**")
                    original_result = adfuller(ts_data.dropna())
                    st.write(f'p-value: {original_result[1]:.4f}')
                    if original_result[1] > 0.05:
                        st.warning("Conclusion: The series is NOT stationary.")
                    else:
                        st.success("Conclusion: The series is stationary.")

                    # Run test on differenced series
                    st.write("**First Differenced Series:**")
                    diff_result = adfuller(ts_data.diff().dropna())
                    st.write(f'p-value: {diff_result[1]:.4f}')
                    if diff_result[1] <= 0.05:
                        st.success("Conclusion: The series is stationary (p <= 0.05).")
                    else:
                        st.warning("Conclusion: The series is NOT stationary (p > 0.05).")

                    st.markdown("#### Model Summary")
                    st.text(str(model.summary()))
                    
                    st.markdown("#### Model Diagnostic Plots")
                    fig_diagnostics = model.plot_diagnostics(figsize=(15, 12))
                    st.pyplot(fig_diagnostics)
                
                # Create and display the forecast
                forecast_result = model.get_forecast(steps=5)
                forecast_mean = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()

                st.subheader("Forecast Visualization")
                fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
                ax_forecast.plot(ts_data, label='Historical Average Prices', marker='o')
                ax_forecast.plot(forecast_mean, label='Forecast (5 Years)', linestyle='--', marker='o')
                ax_forecast.fill_between(conf_int.index,
                                         conf_int.iloc[:, 0],
                                         conf_int.iloc[:, 1],
                                         color='red', alpha=0.1, label='95% Confidence Interval')
                ax_forecast.set_title(f'ARIMA Model Forecast for {city}', fontsize=16)
                ax_forecast.legend(loc='upper left')
                ax_forecast.grid(True)
                st.pyplot(fig_forecast)

                # --- ADDED THIS SECTION FOR YEAR-WISE DETAILS ---
                st.subheader("Year-wise Forecast Details")
                forecast_df = forecast_mean.reset_index()
                forecast_df.columns = ['Year', 'Predicted Average Price']
                forecast_df['Year'] = forecast_df['Year'].dt.year
                st.dataframe(forecast_df.style.format({'Predicted Average Price': '₹{:,.0f}'}))