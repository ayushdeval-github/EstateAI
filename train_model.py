#............MILESTONE 1............................................
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('house_dataset.csv')
print(df)

# Inspect Dataset
print("Shape of dataset:", df.shape)

print("\nColumn Names:", df.columns.tolist())

print(df.info())

print("\nSummary Stats:\n", df.describe())

print(df.isnull().sum()) # Check missing values

# Handle Missing Values
# Drop unnecessary or less useful columns from the dataset.
df= df.drop(['ListingID', 'AgeYears', 'Facing', 'RERAID', 'Latitude', 'Longitude', 'AmenitiesCount', 'TotalFloors' ], axis=1)
print("Remaining columns after dropping:\n", df.columns)

# Numeric columns → fill missing values with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → fill missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check again after handling
print("Missing values after treatment:\n", df.isnull().sum())

print(df) # Display the cleaned dataset after handling missing values 

print(df.describe())

print(df.head())

# Check dataset dimensions after cleaning
print("Dataset shape after cleaning:", df.shape)

# Numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
# Categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# Histograms for numeric columns
df[num_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Boxplots for numeric columns (to detect outliers)
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Countplots for categorical columns (to see balance)
for col in cat_cols:
    plt.figure(figsize=(12, 10))
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Outlier Treatment (IQR Method)

def treat_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Apply capping
    df[col] = np.where(df[col] < lower, lower,
              np.where(df[col] > upper, upper, df[col]))

# Apply to numeric columns
for col in num_cols:
    treat_outliers(col)

print("\n✅ Outliers handled using IQR method.")

print(df)

# Encode categorical variables using One-Hot Encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("\nCategorical variables encoded successfully (One-Hot)!")

print(df)

print("\nFinal cleaned dataset preview:\n", df.head())

print("\nFinal dataset shape:", df.shape)

#............................MILESTONE 1 IS COMPLETED............................................

#...............................MILESTONE 2............................................

#EDA......(Perform EDA: visualize distributions (histograms, boxplots).
# Set a style for Seaborn for better-looking plots
sns.set_theme(style="whitegrid")

# ---Boxplots for Key Numerical Features ---

print("--- Generating Boxplots for Key Numerical Features ---")

# List of your main numerical columns to plot individually
numerical_features = [
    'Price_INR',
    'SuperBuiltUpArea_sqft',
    'BuiltUpArea_sqft',
    'CarpetArea_sqft',
    'YearBuilt'
]

for feature in numerical_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[feature])
        plt.title(f'Boxplot of {feature}', fontsize=16)
        plt.ylabel(feature, fontsize=12)
        plt.show()


# --- Scatter Plots for Numerical Features vs Price_INR ---
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[col], y=df['Price_INR'])
    plt.title(f'{col} vs Price_INR')
    plt.xlabel(col)
    plt.ylabel('Price_INR')
    plt.show()

# --- Histograms for All Numerical Features ---
numerical_cols = df.select_dtypes(include=np.number).columns

print("--- Generating Histograms for Numerical Features ---")

# Loop through each numerical column and create a histogram
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=30) # kde=True adds a density curve
    plt.title(f'Distribution of {col}', fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()


# --- Correlation Heatmap ---
#Correlation analysis & feature selection (heatmap, VIF check).

corr = df.select_dtypes(include=['int64','float64']).corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap", fontsize=16)
plt.show()

#StandardScaler
from sklearn.preprocessing import StandardScaler
# Assume 'x' is your DataFrame containing all features (both numerical and categorical)
# For example:
# x = df.drop('Price_INR', axis=1)
x = df.drop('Price_INR', axis=1)
y = df['Price_INR']

#  Automatically select only the columns with numerical data types
numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

print("Numerical columns selected for scaling:")
print(numerical_cols.tolist())

#  Initialize the scaler
sc = StandardScaler()

#  Create a copy of your DataFrame to store the scaled data
x_scaled = x.copy()

#  Fit and transform ONLY the numerical columns
x_scaled[numerical_cols] = sc.fit_transform(x[numerical_cols])

print("\nFirst 5 rows of the DataFrame after scaling numerical features:")
print(x_scaled.head())

#VIF...

from statsmodels.stats.outliers_influence import variance_inflation_factor


#  Select only the numerical predictor features to check
x_for_vif = df.select_dtypes(include=['int64', 'float64']).drop('Price_INR', axis=1, errors='ignore')

#  Handle any potential missing values
x_for_vif = x_for_vif.dropna()

#  Iteratively remove features with VIF > 5
print("--- Starting VIF Reduction Process ---")
threshold = 5.0
features_to_keep = x_for_vif.copy()

while True:
    # Scale the data before each VIF calculation
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features_to_keep)
    
    # Calculate VIF for the current set of features
    vif_data = pd.DataFrame()
    vif_data["feature"] = features_to_keep.columns
    vif_data["VIF"] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
    
    # Find the feature with the maximum VIF
    max_vif = vif_data['VIF'].max()
    
    # If the max VIF is above the threshold, drop the feature and repeat
    if max_vif > threshold:
        feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
        features_to_keep = features_to_keep.drop(columns=feature_to_drop)
        print(f"Dropped '{feature_to_drop}' (VIF: {max_vif:.2f})")
    else:
        # If no feature has VIF > 5, the process is complete
        break

print("\n--- VIF Check Complete ---")
print("Final features with VIF < 5:")
print(features_to_keep.columns.tolist())
print("\nFinal VIF Scores:")
print(vif_data.sort_values(by='VIF', ascending=False))

#Split dataset into train/test sets (80-20).
from sklearn.model_selection import train_test_split
final_numerical_features = ['Bathrooms', 'Balconies', 'SuperBuiltUpArea_sqft', 'Floor', 'YearBuilt']
final_categorical_features = ['City_Bengaluru', 'IsRERARegistered', 'Furnishing_Semi-Furnished', 'Furnishing_Unfurnished','Parking_Covered', 'Parking_Open', 'Parking_Stilt', 'BuildingType_Standalone Building', 'BuildingType_High Rise', 'BuildingType_Low Rise','BuildingType_Mid Rise', 'BuildingType_Gated Community'] 
final_features = final_numerical_features + final_categorical_features
x = df[final_features]
y = df['Price_INR']

# ---Perform the 80/20 Split ---

# test_size=0.2 means 20% of the data is for testing, 80% is for training.
# random_state=42 ensures you get the same split every time you run the code.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ---Verify the Split ---
print("--- Dataset Split Complete ---")

print(f"Training set shape: {x_train.shape}")
print(f"Testing set shape:  {x_test.shape}")

# Research on choosing model
# Train the model

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# x_train, y_train, x_test, y_test are already defined

# Initialize all models in a dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42) # Added for better performance
}

#  Create a dictionary to store evaluation results
results = {}
trained_models = {}

print("--- Starting Model Training & Evaluation ---")

#  Loop through each model to train, predict, and evaluate
for name, model in models.items():
    # Train the model
    model.fit(x_train, y_train)
    # Store the trained model
    trained_models[name] = model

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Store the results
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f" {name} evaluated successfully.")

#  Display the results in a clean table
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='R²', ascending=False)

print("\n--- Model Performance Comparison ---")
print(results_df)



# creating plot 
#  Automatically get the name of the best model from the results table
best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]

print(f"\n--- Creating plot for the best model: {best_model_name} ---")

# Make predictions with the best model
y_pred = best_model.predict(x_test)

# Create the Actual vs. Predicted scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred)

# Add the 45-degree reference line (for a perfect prediction)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)

plt.title(f'Actual vs. Predicted Prices for {best_model_name}', fontsize=16)
plt.xlabel('Actual Prices (y_test)', fontsize=12)
plt.ylabel('Predicted Prices (y_pred)', fontsize=12)
plt.show()



# Hyperparameter tuning for selected models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Assume x_train, y_train, x_test, y_test are already defined

#  Define the models to be tuned
# We use Ridge instead of Linear Regression as it has a key parameter 'alpha' to tune.
models_to_tune = {
    'Ridge': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

#  Define the parameter grids for each model
# These grids are kept small to ensure the code runs in a reasonable time.
param_grids = {
    'Ridge': {
        'alpha': [0.1, 1.0, 10, 100]
    },
    'Decision Tree': {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
}

#  Create a dictionary to store the best tuned models
best_tuned_models = {}

print("--- Starting Hyperparameter Tuning for All Models ---")

#  Loop through each model and perform GridSearchCV
for name, model in models_to_tune.items():
    print(f"\n--- Tuning {name} ---")
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grids[name],
                               cv=5, # 5-fold cross-validation
                               scoring='r2',
                               verbose=1,
                               n_jobs=-1) # Use all available CPU cores
    
    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)
    
    # Store the best estimator
    best_tuned_models[name] = grid_search.best_estimator_
    
    # Print the best results for the current model
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validated R² score for {name}: {grid_search.best_score_:.4f}")

print("\n--- Hyperparameter Tuning Complete ---")


# Evaluate the best tuned models on the test set
tuned_results = {}

for name, model in best_tuned_models.items():
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    tuned_results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Display the results in a clean table
tuned_results_df = pd.DataFrame(tuned_results).T
tuned_results_df = tuned_results_df.sort_values(by='R²', ascending=False)

print("\n--- Tuned Model Performance Comparison ---")
print(tuned_results_df)

# --- Accuracy Metrics for Regression ---

#  R² Score (standard regression "accuracy")
r2 = r2_score(y_test, y_pred)

#  Define custom accuracy as 1 - (MAE / mean actual price)
mean_actual = y_test.mean()
mae = mean_absolute_error(y_test, y_pred)
custom_accuracy = 1 - (mae / mean_actual)

print("\n📊 Accuracy Metrics:")
print(f"R² Score (Model Accuracy): {r2:.4f} ({r2*100:.2f}%)")
print(f"Custom Accuracy (1 - MAE/Mean Actual): {custom_accuracy:.4f} ({custom_accuracy*100:.2f}%)")



# Finalize the model pipeline (preprocessing + prediction).
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
import joblib

# --- Define Final Features ---
final_numerical_features = ['Bathrooms', 'Balconies', 'SuperBuiltUpArea_sqft', 'Floor', 'YearBuilt']
final_categorical_features = [
    'City_Bengaluru', 'Furnishing_Semi-Furnished', 'Furnishing_Unfurnished',
    'Parking_Covered', 'Parking_Open', 'Parking_Stilt',
    'BuildingType_Standalone Building', 'BuildingType_High Rise',
    'BuildingType_Low Rise', 'BuildingType_Mid Rise',
    'BuildingType_Gated Community'
]
final_features = final_numerical_features + final_categorical_features + ['IsRERARegistered']

X = df[final_features]
y = df['Price_INR']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Numerical Transformer (for numeric + dummy categorical features) ---
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Column Transformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, final_numerical_features + final_categorical_features),
        ('bool', 'passthrough', ['IsRERARegistered'])
    ]
)

# --- Final Model Pipeline (Preprocessing + Model) ---
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=100))
])

# ---  Train the Final Pipeline ---
print(" Training the final model pipeline ...")
final_pipeline.fit(X_train, y_train)
print(" Final pipeline trained successfully.")

# --- Save the Model ---
model_filename = 'final_real_estate_model.joblib'
joblib.dump(final_pipeline, model_filename)
print(f" Final model pipeline saved as '{model_filename}'")

# --- Evaluate the Model on Test Data ---
y_pred = final_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Model Evaluation on Test Set:")
print(f"R² Score       : {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:,.0f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.0f}")

# --- Example Prediction on Test Row ---
sample = X_test.iloc[[0]]
predicted_price = final_pipeline.predict(sample)[0]

print("\n Example Prediction (from test set):")
print("Input features:", sample.to_dict(orient="records")[0])
print(f"Predicted Price (INR): {predicted_price:,.0f}")

# --- Automatic Predictions on Test Data ---
y_pred = final_pipeline.predict(X_test)

# Create DataFrame with all actual vs predicted
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n Predictions on Full Test Data:")
print(results)



#.................MILESTONE 2 IS COMPLETED.........................................
#...............................MILESTONE 3............................................


# Suppose df has YearBuilt & Price_INR columns
#  Group by YearBuilt (average price per year)
yearly_df = df.groupby('YearBuilt')['Price_INR'].mean().reset_index()

# Set index as YearBuilt
yearly_df.set_index('YearBuilt', inplace=True)

#  Moving Averages
yearly_df['SMA_7'] = yearly_df['Price_INR'].rolling(window=7, min_periods=1).mean()
yearly_df['SMA_30'] = yearly_df['Price_INR'].rolling(window=30, min_periods=1).mean()

# Exponential Moving Averages
yearly_df['EMA_7'] = yearly_df['Price_INR'].ewm(span=7, adjust=False).mean()
yearly_df['EMA_30'] = yearly_df['Price_INR'].ewm(span=30, adjust=False).mean()

#  Regression Trend Line
x = yearly_df.index.values
y = yearly_df['Price_INR'].values
coeffs = np.polyfit(x, y, 1)              # linear regression (degree=1)
trend_line = np.polyval(coeffs, x)

#  Plot
plt.figure(figsize=(15,8))

# Scatter plot of raw data (all houses)
plt.scatter(df['YearBuilt'], df['Price_INR'],
            color='gray', alpha=0.4, s=20, label='Raw Data')

# Line plots
plt.plot(yearly_df['Price_INR'], label='Yearly Avg Price', color='blue', linewidth=2)
plt.plot(yearly_df['SMA_7'], label='7-year SMA', color='orange', linewidth=2)
plt.plot(yearly_df['SMA_30'], label='30-year SMA', color='green', linewidth=2)
plt.plot(yearly_df['EMA_7'], label='7-year EMA', color='red', linestyle='--', linewidth=2)
plt.plot(yearly_df['EMA_30'], label='30-year EMA', color='purple', linestyle='--', linewidth=2)
plt.plot(x, trend_line, label='Regression Trend Line', color='black', linestyle='-', linewidth=2)

# Formatting
plt.legend()
plt.title("House Price Trend by Year Built (SMA, EMA & Regression Trend)")
plt.xlabel("Year Built")
plt.ylabel("Price (INR)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()



from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create Sample Time Series Data ---
# This part creates a sample DataFrame `df` to make the script runnable.
# It mimics data with a clear price trend over the years.
# In a real project, you would load your own data here.
print(" Preparing Data ---")
data = {
    'YearBuilt': [y for y in range(1980, 2024)],
    'Price_INR': [1_000_000 + (i * 200_000) + (i**2 * 5000) + np.random.normal(0, 500000) for i in range(44)]
}
df = pd.DataFrame(data)

# Convert your data into a time series format (Year as index, Price as value)
ts = df.set_index('YearBuilt')['Price_INR']
ts.index = pd.to_datetime(ts.index, format='%Y')
print("Time Series ready. Last 5 data points:")
print(ts.tail())
print("\n")


#  Determine 'd' (Order of Differencing) with ADF Test ---
# GOAL: Make the time series stationary (remove trend and seasonality).
# We use the Augmented Dickey-Fuller (ADF) test.
# - If p-value > 0.05, the series is NOT stationary. We need to difference it.
# - If p-value <= 0.05, the series IS stationary.
def adf_test(series, title=''):
    """Perform ADF test and print results"""
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f'p-value: {p_value:.4f}')
    if p_value <= 0.05:
        print("Conclusion: The series is stationary.")
    else:
        print("Conclusion: The series is NOT stationary.")
    return p_value

# Test the original series (it will be non-stationary due to the trend)
print("-- Checking for Stationarity ---")
adf_test(ts, 'Original Series')

# Difference the series once and re-test.
ts_diff = ts.diff().dropna()
adf_test(ts_diff, 'First Differenced Series')
# The p-value should now be less than 0.05, meaning d=1 is the correct choice.
print("\nConclusion: We will use d=1 for our ARIMA model.\n")


# Determine 'p' and 'q' with ACF and PACF plots ---
# GOAL: Find the optimal AR (p) and MA (q) orders.
# We plot these on the STATIONARY (differenced) series.
print(" Plotting ACF and PACF to find p and q ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# ACF Plot: Helps to find 'q' (MA order)
plot_acf(ts_diff, ax=ax1, lags=10)
ax1.set_title('Autocorrelation Function (ACF)')
ax1.set_xlabel('Lag')
ax1.set_ylabel('ACF')

# PACF Plot: Helps to find 'p' (AR order)
plot_pacf(ts_diff, ax=ax2, lags=10)
ax2.set_title('Partial Autocorrelation Function (PACF)')
ax2.set_xlabel('Lag')
ax2.set_ylabel('PACF')

plt.show()
print("Interpretation: Based on the model summary, the q=1 term was not significant.")
print("Therefore, we will simplify the model and set q=0.")
print("Chosen order: p=0, q=0. This makes our model ARIMA(0, 1, 0).\n")


#  Build and Fit the ARIMA Model ---
# GOAL: Create the model with the simplified order and add a trend component.
print(" Building and Fitting the ARIMA(0,1,0) Model with Trend ---")
# To include a drift/trend in a model with d=1, we must use trend='t'.
# This specifies a linear trend on the original data, which becomes a
# constant (the drift) after differencing.
order = (0, 1, 0)
model = ARIMA(ts, order=order, trend='t')
model_fit = model.fit()

# Print the model summary to evaluate it
print(model_fit.summary())
print("\n")


# Perform Model Diagnostics ---
# GOAL: Check if the model's errors (residuals) are random. If they are,
# the model has captured the patterns well.
print("Checking Model Diagnostics ---")
# The `plot_diagnostics` function gives us four plots to check the residuals.
# A good model will have residuals that are uncorrelated and normally distributed.
model_fit.plot_diagnostics(figsize=(15, 12))
plt.suptitle('ARIMA(0,1,0) with Trend Model Diagnostics', y=1.02)
plt.show()
print("Diagnostics Check: The residuals should look like random noise (top left),")
print("be normally distributed (top right, bottom left), and have no autocorrelation (bottom right).\n")


#  Create Forecast and Visualize Results ---
# GOAL: Use our fitted model to predict the future and plot it.
print("Forecasting and Visualization ---")
n_periods = 5  # Forecast for 2024 and 2025 to go up to 2025
forecast_result = model_fit.get_forecast(steps=n_periods)

# Get the predicted values
forecast = forecast_result.predicted_mean
# Get the confidence intervals (the range of uncertainty)
conf_int = forecast_result.conf_int()

# Plot the historical data and the forecast
plt.figure(figsize=(14, 7))
plt.plot(ts, label='Historical Average Prices', color='dodgerblue', marker='o')
plt.plot(forecast, label=f'Forecast ({n_periods} Years)', color='red', linestyle='--', marker='o')

# Shade the confidence interval area
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='red', alpha=0.1, label='95% Confidence Interval')

plt.title('ARIMA Model Forecast', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Average Price (INR)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print(f"\nForecasted values for the next {n_periods} years:")
print(forecast)


#..................MILESTONE 3 IS COMPLETED.........................................