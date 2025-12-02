import pandas as pd
import numpy as np

# --- CONFIGURATION ---
input_file = 'final_dataset.csv'
output_file = 'training_data.csv'

# BIOLOGICAL LAG (The Physics Part)
# Mosquito Lifecycle (Egg to Adult) = 14 days
# Extrinsic Incubation (Virus in Mosquito) = 10 days
# Intrinsic Incubation (Virus in Human) = 6 days
# TOTAL LAG = ~30 to 40 Days
LAG_DAYS = 40 

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print("ERROR: Run clean_data.py first!")
    exit()

df['Date'] = pd.to_datetime(df['Date'])

print(f"Generating Calibrated Data (Lag: {LAG_DAYS} days)...")

simulated_cases = []

for i in range(len(df)):
    # --- THE TIME MACHINE ---
    # Instead of looking at "Today's" weather, we look at weather from 40 days ago.
    # This pushes the peak from July -> September.
    
    if i < LAG_DAYS:
        # For the first few days, we don't have past data, so use Day 0
        past_idx = 0
    else:
        past_idx = i - LAG_DAYS
        
    # Get the "Cause" (Past Weather)
    temp_past = df.loc[past_idx, 'Max_Temp']
    rain_past = df.loc[past_idx, 'Rainfall']
    
    # --- BIOPHYSICAL LOGIC ---
    
    # 1. Temperature Window (Mosquitoes need warm, but not burning hot)
    if temp_past < 18:
        base_cases = 0
    elif temp_past > 33:
        base_cases = 5 # Too hot, they die
    else:
        # Exponential growth in the sweet spot (25-30C)
        base_cases = (temp_past - 18) ** 2.2 
        
    # 2. Rain Accumulation (Breeding Sites)
    # Rain 40 days ago filled the puddles where these mosquitoes were born
    if rain_past > 10.0:
        rain_factor = 1.5 
    elif rain_past > 2.0:
        rain_factor = 1.2
    else:
        rain_factor = 0.9 # Drying out
        
    # 3. Seasonality Multiplier (The "Dashain" Factor)
    # Post-monsoon humidity helps survival. 
    # We add a slight boost if the month is Sept/Oct (Month 9/10)
    current_month = df.loc[i, 'Date'].month
    season_boost = 1.0
    if current_month in [9, 10]: 
        season_boost = 1.5  # The peak multiplier
        
    # --- FINAL CALCULATION ---
    noise = np.random.uniform(0.8, 1.2)
    daily_cases = base_cases * rain_factor * season_boost * noise
    
    # Scaling to realistic Kathmandu numbers (0 to 200 cases/day)
    daily_cases = daily_cases / 2.5
    
    simulated_cases.append(int(daily_cases))

# Add to dataframe
df['Dengue_Cases'] = simulated_cases
df.to_csv(output_file, index=False)

print("-" * 30)
print(f"SUCCESS! Calibrated data saved to '{output_file}'")
print(f"Peak shifted by {LAG_DAYS} days to match Kathmandu's Season.")
print("-" * 30)