import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Real Estate Valuation Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* New Font Import: 'Exo 2' for the title for a cleaner, more technical look */
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@600;700;800&family=Montserrat:wght@400;500;600;700&family=Roboto:wght@400;500&display=swap');

    /* Keyframes for card animations remain the same */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes rotateGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* --- General Body & Background --- */
    body {
        font-family: 'Roboto', sans-serif;
        color: #E0E6F1;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, #1a2a45 0%, #0d1117 35%);
    }

    /* --- Title Styling --- */
    [data-testid="stHeading"] h1 {
        text-align: center !important;
        font-family: 'Exo 2', sans-serif; /* New, cleaner font */
        font-weight: 700;
        font-size: 3.6rem;
        letter-spacing: 0.1em; /* Wider spacing for a premium feel */
        text-transform: uppercase;
        padding: 2.5rem 0 1rem 0; /* Adjusted padding */
        
        /* New premium white/silver gradient */
        background: linear-gradient(90deg, #E0E6F1, #FFFFFF, #E0E6F1);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
        /* Advanced, multi-layered "Aura" effect for a subtle, professional bloom */
        text-shadow: 
            0 0 5px rgba(255, 255, 255, 0.1),  /* Inner soft white glow */
            0 0 15px rgba(88, 166, 255, 0.3),  /* Mid-layer blue ambient glow */
            0 0 35px rgba(163, 113, 247, 0.2); /* Outer soft purple ambient glow */

        animation: fadeInUp 1s ease-out forwards; /* Keep the fade-in animation */
    }

    /* --- Subtitle & Section Headers --- */
    .subtitle {
        text-align: center !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 400; /* Lighter weight for better hierarchy */
        font-size: 1.25rem;
        color: #a0a8b9 !important;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out 0.3s forwards;
        opacity: 0;
    }
    h2 {
        text-align: center !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: #58A6FF;
        margin-top: 3rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out 0.5s forwards;
        opacity: 0;
    }

    /* --- Feature Cards: All styles remain the same --- */
    .feature-card {
        position: relative; background-color: rgba(22, 27, 34, 0.8); border-radius: 15px;
        padding: 2rem; text-align: center; height: 100%; overflow: hidden;
        border: 1px solid #30363D; transition: transform 0.4s ease, box-shadow 0.4s ease;
        backdrop-filter: blur(5px); transform-style: preserve-3d; opacity: 0;
    }
    .card-1 { animation: fadeInUp 1s ease-out 0.7s forwards; }
    .card-2 { animation: fadeInUp 1s ease-out 0.9s forwards; }
    .card-3 { animation: fadeInUp 1s ease-out 1.1s forwards; }
    .card-4 { animation: fadeInUp 1s ease-out 1.3s forwards; }

    .feature-card::before, .feature-card::after, .card-content, .feature-card:hover {
        /* All complex card styles are unchanged */
    }
    .feature-card::before { content: ''; position: absolute; top: 50%; left: 50%; width: 200%; height: 200%; background: conic-gradient(from 0deg, transparent 0%, #58A6FF 20%, #A371F7 50%, #58A6FF 80%, transparent 100%); transform-origin: center center; animation: rotateGlow 4s linear infinite; opacity: 0; transition: opacity 0.4s ease; z-index: 0; }
    .feature-card::after { content: ''; position: absolute; top: 0; left: -150%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent); transform: skewX(-25deg); transition: left 0.75s ease-out; z-index: 2; }
    .card-content { position: relative; z-index: 1; }
    .feature-card h3 { font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 1.5rem; color: #C9D1D9; margin-bottom: 1rem; transition: color 0.3s ease; }
    .feature-card p { font-family: 'Roboto', sans-serif; color: #8B949E; font-size: 1rem; line-height: 1.6; }
    .feature-card:hover { transform: perspective(1000px) rotateX(5deg) rotateY(-5deg) scale(1.05); box-shadow: 0 15px 45px rgba(0, 0, 0, 0.5); border-color: #A371F7; }
    .feature-card:hover::before { opacity: 0.3; }
    .feature-card:hover::after { left: 150%; }
    .feature-card:hover h3 { color: #A371F7; }

    /* Other Custom Styles */
    [data-testid="stAlert"] { background-color: rgba(88, 166, 255, 0.1); border: 1px solid #58A6FF; border-radius: 10px; animation: fadeInUp 1s ease-out 0.6s forwards; opacity: 0; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .footer { text-align: center; padding: 2rem 0; color: #8B949E; font-size: 0.9rem; border-top: 1px solid #30363D; margin-top: 4rem; }
</style>
""", unsafe_allow_html=True)

# --- Main Page Content ---

st.title("AI Real Estate Valuation")
st.markdown('<p class="subtitle">Your Gateway to Intelligent Property Insights</p>', unsafe_allow_html=True)
st.markdown("---") 

st.markdown("<h2>Welcome to the AI-Powered Real Estate Analytics Platform!</h2>", unsafe_allow_html=True)
st.info("This application provides a complete suite of tools for property valuation and market analysis. Use the sidebar to explore different features.", icon="🌟")

st.markdown("<h2>Application Pages</h2>", unsafe_allow_html=True)

# --- Cards remain unchanged ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="feature-card card-1"><div class="card-content"><h3>🔍 Exploratory Data Analysis</h3><p>Explore interactive charts and graphs of the real estate dataset.</p></div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card card-2"><div class="card-content"><h3>📈 Price Trend and Future Forecast</h3><p>Analyze price trends (SMA, EMA, Regression Trend) and forecast with ARIMA + AI narratives.</p></div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card card-3"><div class="card-content"><h3>💰 AI Valuation Dashboard</h3><p>Get AI-based price predictions and run Comparative Market Analysis (CMA).</p></div></div>""", unsafe_allow_html=True)
st.write("") 
left_spacer, col4, right_spacer = st.columns([1, 1.5, 1])
with col4:
    st.markdown("""<div class="feature-card card-4"><div class="card-content"><h3>🤖 AI Real Estate Advisor</h3><p>Ask valuation questions in English, powered by Google's gemini-2.5-flash.</p></div></div>""", unsafe_allow_html=True)
    
st.sidebar.success("✅ Select a tool from the sidebar to begin!")

st.markdown("""<div class="footer"><p> A project by <b>Ayush Deval</b> | AI Real Estate Valuation Hub</p></div>""", unsafe_allow_html=True)