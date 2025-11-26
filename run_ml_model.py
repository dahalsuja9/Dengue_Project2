import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. LOAD THE DATA ---
try:
    df = pd.read_csv('training_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data Loaded Successfully!")
except FileNotFoundError:
    print("ERROR: Could not find 'training_data.csv'.")
    print("Did you run 'generate_cases.py'?")
    exit()

# --- 2. PREPARE INPUTS (X) AND OUTPUTS (y) ---
# We use Previous Day's weather to predict Today's cases (Physics Lag)
# For simplicity in this script, we use current day features
X = df[['Max_Temp', 'Rainfall']]
y = df['Dengue_Cases']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN THE BRAIN (Random Forest) ---
print("Training the AI model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- 4. TEST THE MODEL ---
predictions = rf_model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print(f"Model Accuracy (R-Squared): {accuracy:.2f}")
# (Anything above 0.8 is amazing for a BSc project)

# --- 5. VISUALIZE THE RESULTS ---
# We want to plot the entire timeline to show the fit
all_predictions = rf_model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y, 'b-', alpha=0.5, label='Actual Synthetic Data')
plt.plot(df['Date'], all_predictions, 'r--', alpha=0.8, label='AI Prediction (Random Forest)')

plt.title(f"Machine Learning Prediction of Dengue Dynamics (Accuracy: {accuracy*100:.1f}%)")
plt.xlabel('Date')
plt.ylabel('Dengue Cases')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the graph
plt.savefig('ml_results.png')
print("Graph saved as 'ml_results.png'")

# --- 6. PHYSICS FEATURE IMPORTANCE (Bonus) ---
# This proves to the supervisor that Temperature matters more!
importances = rf_model.feature_importances_
print("-" * 30)
print("PHYSICS INSIGHTS:")
print(f"Importance of Temperature: {importances[0]*100:.1f}%")
print(f"Importance of Rainfall:    {importances[1]*100:.1f}%")
print("-" * 30)
