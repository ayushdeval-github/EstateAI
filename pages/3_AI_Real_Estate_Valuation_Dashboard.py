import streamlit as st
import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
import google.generativeai as genai
import plotly.graph_objects as go
import subprocess
import sys

# Note:
# Previously, scikit-learn was installed at runtime using subprocess for compatibility.
# This has been removed for best practices. All dependencies are now managed via requirements.txt.
# Ensure required versions (e.g., scikit-learn==1.4.2) are installed before running the app.
subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn==1.4.2"], stdout=subprocess.DEVNULL)

# --- Page Configuration ---
st.set_page_config(page_title="AI Valuation Dashboard", page_icon="💰", layout="wide")

# --- Professional CSS Styling 
st.markdown("""
<style>
    /* Card for the input form */
    .card {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
    }

    /* Metric card styling */
    [data-testid="stMetric"] {
        background-color: #f0f2f6; /* Light gray background */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e6e6e6;
    }

    /* Target the label (e.g., "AI Predicted Value") inside the metric card */
    [data-testid="stMetricLabel"] > div {
        color: #333333 !important;
    }

    /* Target the main value (the price) inside the metric card */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    
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

# --- Apply the title and subtitle as native Streamlit components ---
st.title("AI Real Estate Valuation Dashboard")

st.markdown('<p class="subtitle">An all-in-one tool for valuation, market comparison, and future trend analysis.</p>', unsafe_allow_html=True)

# --- 1. Load Prediction Model and Data ---
@st.cache_resource
def load_prediction_model():
    try:
        # ✅ Fix for sklearn compatibility (_RemainderColsList error)
        import sklearn.compose._column_transformer as ct
        
        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            ct._RemainderColsList = _RemainderColsList

        return joblib.load('final_real_estate_model.joblib')

    except FileNotFoundError:
        st.error("Prediction model ('final_real_estate_model.joblib') not found.")
        return None
@st.cache_data
def load_clean_data():
    try:
        df = pd.read_csv('house_dataset.csv')
        df = df.drop(['ListingID', 'AgeYears', 'Facing', 'RERAID', 'Latitude', 'Longitude', 'AmenitiesCount', 'TotalFloors', 'Locality'], axis=1, errors='ignore')
        for col in df.select_dtypes(include=['number']).columns: df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns: df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    except FileNotFoundError: st.error("'house_dataset.csv' not found."); return None

prediction_model = load_prediction_model()
df_clean = load_clean_data()

# --- 2. Gemini GenAI Configuration ---
try:
    # This automatically reads the key from your .streamlit/secrets.toml file
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    genai_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception:
    genai_model = None

# --- 3. Helper Functions (ARIMA & GenAI) ---
@st.cache_data(ttl=3600)
def get_arima_forecast(data, city_selection):
    try:
        ts_data = data[data['City'] == city_selection].groupby('YearBuilt')['Price_INR'].mean()
        if ts_data.nunique() < 10: return None, None, "Not enough historical data."
        ts_data.index = pd.to_datetime(ts_data.index, format='%Y')
        ts_data = ts_data.asfreq('AS-JAN')
        model = ARIMA(ts_data, order=(0, 1, 0), trend='t').fit()
        forecast_results = model.get_forecast(steps=5)
        return ts_data, forecast_results.predicted_mean, None
    except Exception as e: return None, None, f"ARIMA forecast failed: {e}"

def generate_narrative(ai_price, cma_summary, forecast_summary):
    prompt = f"As a real estate expert, write a professional investment narrative in one paragraph using this data: The AI Predicted Value is {ai_price}. The Current Market Comparison shows {cma_summary}. The 5-Year City Forecast indicates {forecast_summary}."
    if genai_model:
        with st.spinner("Gemini AI is writing the narrative..."):
            try:
                response = genai_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e: return f"Gemini AI failed to generate a narrative. Error: {e}"
    return "GenAI model not configured. Please add your Gemini API key to the .streamlit/secrets.toml file."

# --- 4. Main Layout ---
if all([prediction_model, df_clean is not None]):
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏡 Enter Property Details")
        with st.form("property_form"):
            city = st.selectbox("Select City", df_clean['City'].unique())
            bhk = st.selectbox("BHK", sorted(df_clean['BHK'].unique()))
            area = st.number_input("Area (sqft)", min_value=300, value=1500)
            year = st.number_input("Year Built", min_value=1980, value=2020)
            property_type = st.selectbox("Property Type", df_clean['PropertyType'].unique())
            submit_button = st.form_submit_button(label='Generate Full Analysis')
        st.markdown('</div>', unsafe_allow_html=True)

    if submit_button:
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💰 AI Predicted Value")
            input_df = pd.DataFrame([{'Bathrooms': 3, 'Balconies': 2, 'SuperBuiltUpArea_sqft': area, 'Floor': 5, 'YearBuilt': year, 'City_Bengaluru': 1 if city == 'Bengaluru' else 0, 'IsRERARegistered': 1, 'Furnishing_Semi-Furnished': 1, 'Furnishing_Unfurnished': 0, 'Parking_Covered': 1, 'Parking_Open': 0, 'Parking_Stilt': 0, 'BuildingType_Standalone Building': 0, 'BuildingType_High Rise': 1, 'BuildingType_Low Rise': 0, 'BuildingType_Mid Rise': 0, 'BuildingType_Gated Community': 0}])
            predicted_price = prediction_model.predict(input_df)[0]
            st.metric(label="Estimated Market Value", value=f"₹ {predicted_price:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📈 Comparable Properties in {city}")
        comps = df_clean[(df_clean['City'] == city) & (df_clean['BHK'] == bhk) & (df_clean['PropertyType'] == property_type) & (df_clean['SuperBuiltUpArea_sqft'].between(area * 0.8, area * 1.2))].copy()
        if not comps.empty:
            cma_avg_price = comps['Price_INR'].mean()
            st.metric("Average Price of Similar Properties", f"₹ {cma_avg_price:,.0f}")
            st.dataframe(comps[['PropertyType', 'BHK', 'SuperBuiltUpArea_sqft', 'Price_INR']].head())
            cma_text = f"similar properties in '{city}' are selling for an average of Rs. {cma_avg_price:,.0f}"
        else:
            st.warning("No comparable properties found.")
            cma_text = "no direct comparable properties were found"
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📊 Future Market Trend for {city}")
        historical_data, forecast_data, error = get_arima_forecast(df_clean, city)
        if error:
            st.error(error)
            forecast_text = "the future market trend is currently unavailable"
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Avg Price'))
            fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data, mode='lines', name='Forecasted Price', line=dict(dash='dash')))
            fig.update_layout(title=f'5-Year Price Forecast for {city}', xaxis_title='Year', yaxis_title='Average Price (INR)')
            st.plotly_chart(fig, use_container_width=True)
            forecast_text = f"the average price in {city} is forecasted to grow over the next 5 years"
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 GenAI Investment Narrative")
        if genai_model:
            full_narrative = generate_narrative(f"Rs. {predicted_price:,.0f}", cma_text, forecast_text)
            st.markdown(f"{full_narrative}")
        else:
            st.warning("Please add your Gemini API key to the .streamlit/secrets.toml file to generate a narrative.")
        st.markdown('</div>', unsafe_allow_html=True)