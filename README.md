# 🏡 EstateAI – AI Real Estate Valuation System

🚀 **EstateAI** is an intelligent, multi-page Streamlit application that predicts property prices, analyzes market trends, and generates AI-powered investment insights for real estate decision-making.

---

## 🌟 Live Demo


`https://estateaiapp.streamlit.app/`

---

## 🚀 Key Features

### 💰 AI Property Valuation

* Predicts real estate prices using trained Machine Learning models
* Supports dynamic user inputs (city, area, BHK, etc.)

### 📊 Comparable Market Analysis (CMA)

* Displays similar properties based on filters
* Provides average pricing insights

### 📈 Future Market Forecasting

* Uses **ARIMA time-series model**
* Predicts price trends for next 5 years

### 🤖 AI Real Estate Advisor

* Generates professional investment insights
* Powered by **Google Gemini AI**

### 📉 Exploratory Data Analysis

* Visual insights into dataset
* Distribution, trends, and correlations

---

## 🧠 Tech Stack

| Category       | Technology           |
| -------------- | -------------------- |
| Frontend       | Streamlit            |
| Backend        | Python               |
| ML Models      | Scikit-learn         |
| Time Series    | Statsmodels (ARIMA)  |
| Visualization  | Plotly               |
| AI Integration | Google Generative AI |
| Model Storage  | Joblib               |

---

## 📂 Project Structure

```bash
EstateAI/
│── home.py
│── pages/
│   ├── 1_Exploratory_Data_Analysis.py
│   ├── 2_Price_Trend_and_Future_Forecast.py
│   ├── 3_AI_Real_Estate_Valuation_Dashboard.py
│   ├── 4_AI_Real_Estate_Advisor.py
│── train_model.py
│── final_real_estate_model.joblib
│── house_dataset.csv
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ayushdeval-github/EstateAI.git
cd EstateAI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

Create file:

```bash
.streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the Application

```bash
streamlit run home.py
```

---

## 🌐 Deployment

You can deploy using:

* ✅ **Streamlit Cloud**
* ✅ **Hugging Face Spaces**

---

## 📊 Model Details

### Algorithms Used

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Ridge Regression

### Final Model

* Optimized pipeline with preprocessing + regression
* Saved using `joblib`

### Evaluation Metrics

* R² Score
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

## 📈 Future Enhancements

* 🌍 Multi-city / global dataset support
* 📍 Location-based prediction (lat/long)
* 🧠 Deep learning models
* 📱 Mobile responsive UI
* 🗺 Map integration

---

## 👨‍💻 Author

**Ayush Deval**
🎓 Computer Science & Engineering

---

## ⭐ Support

If you like this project:

* ⭐ Star this repository
* 🍴 Fork it
* 📢 Share it

---

## 📌 Note

* Ensure correct library versions for compatibility
* Model is version-dependent on `scikit-learn==1.6.1`

---
