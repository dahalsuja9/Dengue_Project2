import pandas as pd
import os

# --- FILE NAMES ---
input_file = 'nepal_dengue_data.csv'
output_file = 'final_dataset.csv'

print(f"Reading {input_file}...")

try:
    # 1. Load the CSV
    # keeping skiprows=10 since it successfully found the header row last time
    df = pd.read_csv(input_file, skiprows=10)

    # 2. Check columns
    print("Columns found:", df.columns.tolist())
    
    # 3. Create 'Date' from YEAR and DOY (Day of Year)
    # The format '%Y%j' tells Python that the number is "Year" + "Day of Year"
    print("Converting Day-of-Year to standard Date...")
    df['Date'] = pd.to_datetime(df['YEAR'] * 1000 + df['DOY'], format='%Y%j')

    # 4. Select and Rename columns
    # We rename 'T2M' -> 'Max_Temp' and 'PRECTOTCORR' -> 'Rainfall'
    df = df.rename(columns={
        'T2M': 'Max_Temp', 
        'PRECTOTCORR': 'Rainfall'
    })

    # 5. Add the empty 'Dengue_Cases' column
    df['Dengue_Cases'] = 0

    # 6. Reorder and Save
    # We only want these 4 final columns
    final_df = df[['Date', 'Max_Temp', 'Rainfall', 'Dengue_Cases']]
    final_df.to_csv(output_file, index=False)

    print("-" * 30)
    print(f"SUCCESS! Created '{output_file}'")
    print("-" * 30)
    print("NEXT STEP: Open 'final_dataset.csv' in LibreOffice and fill in the Dengue Cases.")

except Exception as e:
    print("\nERROR:", e)
    print("If it says 'skiprows', try changing the number 10 in the script to 11 or 12.")
