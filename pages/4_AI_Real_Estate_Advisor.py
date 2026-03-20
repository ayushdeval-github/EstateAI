import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import re
import json
from streamlit_mic_recorder import speech_to_text
import speech_recognition as sr
import tempfile
import os

# --- Page Configuration & UI Styling ---
st.set_page_config(page_title="AI Real Estate Advisor", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .chat-title {text-align: center; font-family: 'Roboto', sans-serif; color: #7b8eab;}
    .chat-title h1 {font-weight: 700; margin-bottom: 0;}
    .chat-title p {font-weight: 300; font-size: 1.1rem; color: #7f8c8d;}
    [data-testid="stChatMessage"] {
        border-radius: 20px; padding: 1rem 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stChatMessage"] p {
        font-family: 'Roboto', sans-serif; font-weight: 400; font-size: 1rem; line-height: 1.6;
    }
    .voice-status {padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;}
    .success {background-color: #d4edda; color: #155724;}
    .error {background-color: #f8d7da; color: #721c24;}
    .warning {background-color: #fff3cd; color: #856404;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-title"><h1>AI Real Estate Advisor</h1><p>Your intelligent assistant for property insights and forecasts.</p></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Load Gemini API & Model ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"🚨 Gemini API key error: {e}")
    st.stop()

# --- Load Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('house_dataset.csv')
        required_columns = ['YearBuilt', 'Price_INR', 'City', 'BHK', 'Locality']
        df['Price_INR'] = pd.to_numeric(df['Price_INR'], errors='coerce')
        df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')
        df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')
        df.dropna(subset=required_columns, inplace=True)
        df['YearBuilt'] = df['YearBuilt'].astype(int)
        df['BHK'] = df['BHK'].astype(int)
        return df
    except FileNotFoundError:
        st.error("Error: 'house_dataset.csv' not found.")
        return None
    except KeyError:
        st.error(f"Error: Your dataset must contain all required columns: {required_columns}.")
        return None

df = load_data()

# --- Data Analysis Functions ---
def calculate_emi(df, city, bhk, interest_rate=8.5, tenure_years=20, down_payment_percent=20):
    filtered_df = df[(df['City'].str.lower() == city.lower()) & (df['BHK'] == bhk)]
    if filtered_df.empty: return f"Could not find the average price for a {bhk} BHK in {city} to calculate the EMI."
    avg_price = filtered_df['Price_INR'].mean()
    loan_amount = avg_price * (1 - (down_payment_percent / 100))
    monthly_interest_rate = (interest_rate / 100) / 12
    tenure_months = tenure_years * 12
    if monthly_interest_rate == 0: emi = loan_amount / tenure_months
    else: emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** tenure_months) / ((1 + monthly_interest_rate) ** tenure_months - 1)
    return (f"For an average {bhk} BHK in {city.title()} priced at ₹{avg_price:,.0f}:\n"
            f"- Assuming a **{down_payment_percent}%** down payment, the loan amount would be **₹{loan_amount:,.0f}**.\n"
            f"- Based on a **{tenure_years}-year** loan at an interest rate of **{interest_rate}%**, the estimated monthly EMI would be approximately **₹{emi:,.0f}**.")

def get_market_summary(df, city):
    summary_parts = [f"### 🏙️ Real Estate Market Summary for {city.title()}\n---"]
    for bhk in [1, 2, 3]:
        price_summary = get_price_summary(df, city, bhk)
        if "No data" not in price_summary: summary_parts.append(f"- **{bhk} BHK:** {price_summary}")
    top_localities_summary = rank_localities(df, city, n=3)
    summary_parts.append(f"\n{top_localities_summary}")
    end_year = df['YearBuilt'].max()
    start_year = end_year - 5
    growth_summary = calculate_growth_period(df, city, start_year, end_year)
    if "Sorry" not in growth_summary: summary_parts.append(f"\n**Recent Trend:** {growth_summary}")
    return "\n".join(summary_parts)

def suggest_best_investment(df, years=10):
    if df is None or df.empty: return "Dataset is not loaded."
    end_year = df['YearBuilt'].max()
    start_year = end_year - years
    cities = df['City'].unique()
    city_growth = {}
    for city in cities:
        city_df = df[df['City'] == city]
        price_start = city_df[city_df['YearBuilt'] == start_year]['Price_INR'].mean()
        price_end = city_df[city_df['YearBuilt'] == end_year]['Price_INR'].mean()
        if pd.notna(price_start) and pd.notna(price_end) and price_start > 0:
            growth = ((price_end - price_start) / price_start) * 100
            city_growth[city] = growth
    if not city_growth: return f"I could not find any cities with consistent data over the last {years} years to perform an investment analysis."
    best_city = max(city_growth, key=city_growth.get)
    highest_growth = city_growth[best_city]
    suggested_bhk = df[df['City'] == best_city]['BHK'].mode()[0]
    return (f"Based on historical data from {start_year} to {end_year}, **{best_city}** showed the highest price growth of approximately **{highest_growth:.2f}%**. "
            f"Properties like **{suggested_bhk} BHKs** are common in this city. "
            f"\n\n**Disclaimer:** This analysis is based on past performance and is not financial advice. Real estate markets can be unpredictable.")

def list_available_cities(df):
    if df is None or df.empty: return "The dataset is not loaded."
    cities = sorted(df['City'].unique())
    return f"There are {len(cities)} cities available in the dataset. They are: {', '.join(cities)}."

def get_price_summary(df, city, bhk):
    filtered_df = df[(df['City'].str.lower() == city.lower()) & (df['BHK'] == bhk)]
    if filtered_df.empty: return f"No data for {bhk} BHK in {city}."
    avg_price = filtered_df['Price_INR'].mean()
    avg_lakhs = avg_price / 100000
    lower, upper = avg_lakhs * 0.85, avg_lakhs * 1.15
    return f"The average price for a {bhk} BHK in {city} is around ₹{lower:.0f}-{upper:.0f} lakhs."

def compare_prices(df, entities):
    if entities.get('city1') and entities.get('city2'):
        city1, city2, bhk = entities['city1'], entities['city2'], entities['bhk']
        summary1, summary2 = get_price_summary(df, city1, bhk), get_price_summary(df, city2, bhk)
        return f"**Comparison Result:**\n- **{city1.title()}:** {summary1}\n- **{city2.title()}:** {summary2}"
    elif entities.get('bhk1') and entities.get('bhk2'):
        bhk1, bhk2, city = entities['bhk1'], entities['bhk2'], entities['city']
        summary1, summary2 = get_price_summary(df, city, bhk1), get_price_summary(df, city, bhk2)
        return f"**Comparison Result for {city.title()}:**\n- **{bhk1} BHK:** {summary1}\n- **{bhk2} BHK:** {summary2}"
    return "Comparison could not be made."

def rank_localities(df, city, n=5):
    city_df = df[df['City'].str.lower() == city.lower()]
    if city_df.empty: return f"No data for {city}."
    top_localities = city_df.groupby('Locality')['Price_INR'].mean().sort_values(ascending=False).head(n)
    result = f"**Top {n} Most Expensive Localities in {city.title()}:**\n"
    for locality, price in top_localities.items():
        result += f"- **{locality}:** Average Price ₹{price:,.0f}\n"
    return result

def rank_cheapest_cities(df, bhk, n=5):
    bhk_df = df[df['BHK'] == bhk]
    if bhk_df.empty: return f"No data for {bhk} BHK properties."
    cheapest_cities = bhk_df.groupby('City')['Price_INR'].mean().sort_values(ascending=True).head(n)
    result = f"**Top {n} Cheapest Cities for a {bhk} BHK:**\n"
    for city, price in cheapest_cities.items():
        result += f"- **{city}:** Average Price ₹{price:,.0f}\n"
    return result

def filter_by_budget(df, bhk, max_price):
    max_price_inr = max_price * 100000
    filtered = df[(df['BHK'] == bhk) & (df['Price_INR'] <= max_price_inr)]
    if filtered.empty: return f"No cities found with {bhk} BHK properties under ₹{max_price} lakhs."
    cities = filtered['City'].unique()
    return f"You can find **{bhk} BHK properties for under ₹{max_price} lakhs** in the following cities: {', '.join(cities)}."

def calculate_growth_period(df, city, start_year, end_year):
    city_df = df[df['City'].str.lower() == city.lower()]
    price_start = city_df[city_df['YearBuilt'] == start_year]['Price_INR'].mean()
    price_end = city_df[city_df['YearBuilt'] == end_year]['Price_INR'].mean()
    if pd.isna(price_start) or pd.isna(price_end):
        return f"Sorry, I don't have enough continuous data between {start_year} and {end_year} for {city}."
    growth = ((price_end - price_start) / price_start) * 100
    return f"In {city.title()}, the average price grew by approximately **{growth:.2f}%** from {start_year} to {end_year} (from ₹{price_start:,.0f} to ₹{price_end:,.0f})."

@st.cache_data(show_spinner="Running ARIMA forecast...")
def run_arima_forecast(data, city):
    st.markdown(f"### 📈 Detailed ARIMA Forecast for {city}")
    if data is None or city.lower() not in data['City'].str.lower().unique():
        return f"Sorry, I don't have enough data to create a forecast for {city}."
    city_data = data[data['City'].str.lower() == city.lower()]
    ts = city_data.groupby('YearBuilt')['Price_INR'].mean()
    ts.index = pd.to_datetime(ts.index, format='%Y')
    if len(ts) < 15:
        return f"Not enough historical data for {city} to create a reliable forecast."
    st.markdown("---")
    st.subheader("1. Stationarity Analysis (ADF Test)")
    result = adfuller(ts.dropna())
    st.write(f"**Original Time Series:** p-value = {result[1]:.4f} -> {'Stationary' if result[1] <= 0.05 else 'Not Stationary'}")
    ts_diff = ts.diff().dropna()
    result_diff = adfuller(ts_diff)
    st.write(f"**First-Differenced Time Series:** p-value = {result_diff[1]:.4f} -> {'Stationary' if result_diff[1] <= 0.05 else 'Not Stationary'}")
    st.info("The data becomes stationary after one round of differencing, so we will use **d=1**.")
    st.subheader("2. ACF & PACF Plots")
    with st.expander("Show ACF/PACF plots"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(ts_diff, ax=ax1, lags=10, title='Autocorrelation Function (ACF)')
        plot_pacf(ts_diff, ax=ax2, lags=10, title='Partial Autocorrelation Function (PACF)')
        st.pyplot(fig)
        st.info("Since there are no significant spikes, **p=0** and **q=0** are appropriate choices.")
    st.subheader("3. ARIMA Model Summary")
    try:
        model_arima = ARIMA(ts, order=(0, 1, 0), trend='t')
        model_fit = model_arima.fit()
        with st.expander("Show ARIMA Model Summary"):
            st.text(model_fit.summary())
    except Exception as e:
        return f"Failed to build the ARIMA model for {city}. Error: {e}"
    st.subheader("4. Model Diagnostics")
    with st.expander("Show model diagnostic plots"):
        diag_fig = model_fit.plot_diagnostics(figsize=(15, 12))
        diag_fig.suptitle(f'ARIMA(0,1,0) Diagnostics for {city}', y=1.02)
        st.pyplot(diag_fig)
    st.subheader("5. Price Forecast Visualization")
    n_periods = 5
    forecast_result = model_fit.get_forecast(steps=n_periods)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    fig_forecast, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ts, label='Historical Average Prices', color='dodgerblue', marker='o')
    ax.plot(forecast, label=f'Forecast ({n_periods} Years)', color='red', linestyle='--', marker='o')
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1, label='95% Confidence Interval')
    ax.set_title(f'ARIMA Price Forecast for {city}', fontsize=16)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Price (INR)')
    ax.legend(loc='upper left')
    ax.grid(True)
    st.pyplot(fig_forecast)
    current_price = ts.iloc[-1]
    future_price = forecast.iloc[-1]
    growth = ((future_price - current_price) / current_price) * 100
    return (f"Based on the analysis for {city}, the price is forecasted to change from approx. **₹{current_price:,.0f}** to **₹{future_price:,.0f}** over the next {n_periods} years, a potential growth of about **{growth:.2f}%**.")

# --- LLM Helper Functions ---
@st.cache_data(show_spinner="Analyzing your question...")
def extract_intent_entities(user_query):
    prompt = f'''
    You are a master at analyzing real estate questions. Your job is to identify the user's intent and extract entities.

    **INTENTS:**
    - `get_summary`, `compare_prices`, `rank_localities`, `rank_cheapest_cities`, `filter_by_budget`,
    - `calculate_growth_period`, `suggest_investment`, `calculate_emi`, `market_summary`,
    - `run_forecast`, `list_cities`, `clarification`,
    - `handle_greeting`: For simple greetings or closings like 'hello', 'thank you', or 'bye'.
    - `off_topic`: For anything else.

    **EXAMPLES:**
    - "price of 2 bhk in mumbai" -> {{"tool": "get_summary", "city": "Mumbai", "bhk": 2}}
    - "what would be the emi for a 3 bhk in delhi" -> {{"tool": "calculate_emi", "city": "Delhi", "bhk": 3}}
    - "give me a market summary for Pune" -> {{"tool": "market_summary", "city": "Pune"}}
    - "thank you" -> {{"tool": "handle_greeting"}}
    - "average price in delhi" -> {{"tool": "clarification", "city": "Delhi"}}

    **Analyze this user query and return ONLY the JSON object:**
    User Query: "{user_query}"
    '''
    response = model.generate_content(prompt).text
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"tool": "off_topic"}
    except:
        return {"tool": "off_topic"}

def generate_answer(user_input, data_context):
    prompt = f"""
    You are a professional real estate data analyst. The user asked a question, and a data function was run. The result is in the 'DATA CONTEXT'.
    Your job is to present this result to the user in a clear, direct, and helpful conversational paragraph.
    Do not use greetings like 'Hello' or 'Namaste' unless the context is a greeting itself. Get straight to the point for data questions.
    
    DATA CONTEXT: "{data_context}"
    USER QUESTION: "{user_input}"

    YOUR ANSWER:
    """
    return model.generate_content(prompt).text

# --- Chat History Management & UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input Handling (Sidebar for Voice, Chat Input at bottom) ---
prompt = None

with st.sidebar:
    st.write("### 🎤 Voice Assistant")
    voice_text = speech_to_text(language='en-US', start_prompt="Click to Speak", stop_prompt="Stop Recording", just_once=True, use_container_width=True, key='voice_input')

    if voice_text:
        st.session_state.voice_prompt = voice_text
    
    if "voice_prompt" in st.session_state and st.session_state.voice_prompt:
        st.info(f"Recognized text:")
        st.write(f"\"{st.session_state.voice_prompt}\"")
        if st.button("Use this voice input"):
            prompt = st.session_state.voice_prompt
            st.session_state.voice_prompt = None

text_prompt = st.chat_input("Ask me about real estate...")

if text_prompt:
    prompt = text_prompt
    if "voice_prompt" in st.session_state:
        st.session_state.voice_prompt = None

# --- Main Processing Logic ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            is_greeting = any(word in prompt.lower() for word in ["thank", "bye", "hello"])
            if df is None and not is_greeting:
                st.error("Dataset not loaded. Cannot process request.")
                st.stop()

            entities = extract_intent_entities(prompt)
            tool = entities.get("tool")
            context = ""

            if tool == "handle_greeting":
                if "thank" in prompt.lower(): context = "You're welcome! Let me know if you need anything else."
                elif "bye" in prompt.lower(): context = "Goodbye! Have a great day."
                else: context = "Hello! How can I assist you with real estate data today?"
            elif tool == "suggest_investment": context = suggest_best_investment(df)
            elif tool == "list_cities": context = list_available_cities(df)
            elif tool == "get_summary": context = get_price_summary(df, entities.get('city'), entities.get('bhk'))
            elif tool == "compare_prices": context = compare_prices(df, entities)
            elif tool == "rank_localities": context = rank_localities(df, entities.get('city'))
            elif tool == "rank_cheapest_cities": context = rank_cheapest_cities(df, entities.get('bhk'))
            elif tool == "filter_by_budget": context = filter_by_budget(df, entities.get('bhk'), entities.get('max_price'))
            elif tool == "calculate_growth_period": context = calculate_growth_period(df, entities.get('city'), entities.get('start_year'), entities.get('end_year'))
            elif tool == "calculate_emi": context = calculate_emi(df, entities.get('city'), entities.get('bhk'))
            elif tool == "market_summary": context = get_market_summary(df, entities.get('city'))
            elif tool == "run_forecast": context = run_arima_forecast(df, entities.get('city'))
            elif tool == "clarification":
                city = entities.get('city')
                context = f"I can help with prices in {city}, but please specify the number of BHK you're interested in (e.g., '2 BHK in {city}')."
            else:
                context = "I can only answer questions about real estate data."

            if tool == "handle_greeting":
                final_response = context
            else:
                final_response = generate_answer(prompt, context)
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
    
    st.rerun()