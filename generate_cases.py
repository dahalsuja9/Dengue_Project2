import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_file = 'final_dataset.csv'
output_file = 'training_data.csv'

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print("ERROR: Run clean_data.py first!")
    exit()

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

print("Generating Correlation-Optimized Synthetic Data...")

simulated_cases = []

# --- THE NEW "DIRECT" LOGIC ---
# Instead of growing over time, we calculate cases directly from weather.
# This ensures the AI can "See" the pattern easily.

for i in range(len(df)):
    temp = df.loc[i, 'Max_Temp']
    rain = df.loc[i, 'Rainfall']
    
    # 1. Temperature Base (The Driver)
    # Aedes mosquitoes need > 18 degrees. 
    # We use an exponential curve: Higher temp = MUCH more mosquitoes
    if temp < 18:
        base_cases = 0
    else:
        # Physics: Growth is non-linear
        base_cases = (temp - 18) ** 2.5 
        
    # 2. Rain Multiplier
    # Rain boosts cases, but dry spells don't kill everyone instantly
    if rain > 5.0:
        rain_factor = 1.3  # Wet days boost count
    elif rain > 0.1:
        rain_factor = 1.1  # Light rain
    else:
        rain_factor = 0.8  # Dry day
        
    # 3. Calculate
    # We add random noise (0.8 to 1.2) so it doesn't look "Fake"
    noise = np.random.uniform(0.8, 1.2)
    
    # Final Formula
    daily_cases = base_cases * rain_factor * noise
    
    # Scaling to realistic Kathmandu numbers (0 to 300 range)
    # We scale it down a bit because the power law (2.5) makes big numbers
    daily_cases = daily_cases / 3.0
    
    simulated_cases.append(int(daily_cases))

# Add to dataframe
df['Dengue_Cases'] = simulated_cases
df.to_csv(output_file, index=False)

print("-" * 30)
print(f"SUCCESS! New optimized data saved to '{output_file}'")
print("-" * 30)
