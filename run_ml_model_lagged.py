import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. CONFIGURATION ---
LAG_DAYS = 40  # This MUST match the lag in your generator script
INPUT_FILE = 'training_data.csv'
OUTPUT_GRAPH = 'ml_results_lagged.png' # New name to avoid overwriting

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded '{INPUT_FILE}' successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{INPUT_FILE}'.")
    exit()

# --- 3. FEATURE ENGINEERING (The "Smart" Part) ---
print(f"Creating Lag Features (Looking back {LAG_DAYS} days)...")

# Create new columns representing past weather
df['Max_Temp_Lag'] = df['Max_Temp'].shift(LAG_DAYS)
df['Rainfall_Lag'] = df['Rainfall'].shift(LAG_DAYS)

# Drop the first 40 days (which now have empty/NaN data)
df_clean = df.dropna().copy()

# --- 4. PREPARE INPUTS ---
# Train on the PAST weather (Lagged columns), predict TODAY's Cases
X = df_clean[['Max_Temp_Lag', 'Rainfall_Lag']]
y = df_clean['Dengue_Cases']

# Split 80% Training / 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. TRAIN MODEL ---
print("Training Random Forest on Lagged Data...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 6. TEST & SCORE ---
predictions = rf_model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print("-" * 30)
print(f"FINAL MODEL ACCURACY (R²): {accuracy:.4f}")
print("-" * 30)

# --- 7. VISUALIZE ---
# Plot the whole timeline to show the fit
all_predictions = rf_model.predict(X)

plt.figure(figsize=(12, 6))
# Plot Actual Data
plt.plot(df_clean['Date'], y, color='blue', alpha=0.4, label='Actual Data (Simulated)')
# Plot AI Prediction
plt.plot(df_clean['Date'], all_predictions, color='red', linestyle='--', alpha=0.8, label='AI Prediction (With Lag)')

plt.title(f"AI Prediction with {LAG_DAYS}-Day Biological Lag (Accuracy: {accuracy*100:.1f}%)")
plt.xlabel('Date')
plt.ylabel('Dengue Cases')
plt.legend()
plt.grid(True, alpha=0.3)

# Save as a NEW image file
plt.savefig(OUTPUT_GRAPH)
print(f"Graph saved as '{OUTPUT_GRAPH}'")

# --- 8. PHYSICS INSIGHTS ---
importances = rf_model.feature_importances_
print("\nPHYSICS DRIVERS:")
print(f"Importance of Temperature (-{LAG_DAYS} days): {importances[0]*100:.1f}%")
print(f"Importance of Rainfall    (-{LAG_DAYS} days): {importances[1]*100:.1f}%")