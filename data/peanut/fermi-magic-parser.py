import re
import numpy as np
import pandas as pd

# 1. Define the raw table data as a string
with open("fermi-lat-magic.txt", "r") as f:
    raw_data = f.read()

# 2. Function to parse individual measurement strings
def parse_measurement(val_str):
    val_str = val_str.strip()
    if not val_str or val_str == 'cdots':
        return None
    
    # Check if it is an upper limit (< value x 10^exp)
    if val_str.startswith('<'):
        match = re.match(r'<([\d.]+)\s*x\s*10\^([-\d]+)', val_str)
        if match:
            base, exp = match.groups()
            flux = float(base) * (10**int(exp))
            flux_err = 0.2 * flux
            uplims = True
            return flux, flux_err, uplims
            
    # Check if it is a regular data point with errors ((base +or- err) x 10^exp)
    elif '+or-' in val_str:
        match = re.match(r'\(([\d.]+)\s*\+or-\s*([\d.]+)\)\s*x\s*10\^([-\d]+)', val_str)
        if match:
            base, err, exp = match.groups()
            multiplier = 10**int(exp)
            flux = float(base) * multiplier
            flux_err = float(err) * multiplier
            uplims = False
            return flux, flux_err, uplims
            
    return None

# 3. Process the table row by row
fermi_rows = []
magic_rows = []

for line in raw_data.strip().split('\n'):
    # Split using tab character and clean spaces
    cols = [c.strip() for c in line.split('\t')]
    # Pad columns to ensure 6 positions (3 for left layout, 3 for right layout)
    while len(cols) < 6:
        cols.append('')
        
    # --- Process Left Columns ---
    # Layout: [0]=Energy, [1]=Fermi-LAT, [2]=MAGIC
    e_left = cols[0]
    if e_left:
        energy = float(e_left)
        # Parse Fermi-LAT
        fermi_res = parse_measurement(cols[1])
        if fermi_res:
            fermi_rows.append((energy, *fermi_res))
        # Parse MAGIC
        magic_res = parse_measurement(cols[2])
        if magic_res:
            magic_rows.append((energy, *magic_res))
            
    # --- Process Right Columns ---
    # Layout: [3]=Energy, [4]=Fermi-LAT, [5]=MAGIC
    e_right = cols[3]
    if e_right:
        energy = float(e_right)
        # Parse Fermi-LAT
        fermi_res = parse_measurement(cols[4])
        if fermi_res:
            fermi_rows.append((energy, *fermi_res))
        # Parse MAGIC
        magic_res = parse_measurement(cols[5])
        if magic_res:
            magic_rows.append((energy, *magic_res))

# 4. Build dataframes and scale values to units of  TeV cm^-2 s^-1
df_fermi = pd.DataFrame(fermi_rows, columns=['Energy', 'Flux', 'FluxErr', 'Uplims'])
df_fermi['Flux'] = df_fermi['Flux']
df_fermi['FluxErr'] = df_fermi['FluxErr']

df_magic = pd.DataFrame(magic_rows, columns=['Energy', 'Flux', 'FluxErr', 'Uplims'])
df_magic['Flux'] = df_magic['Flux']
df_magic['FluxErr'] = df_magic['FluxErr']

# 5. Sort tables by energy
df_fermi = df_fermi.sort_values('Energy').reset_index(drop=True)
df_magic = df_magic.sort_values('Energy').reset_index(drop=True)

# 6. Output / Save Results
print("--- Fermi-LAT Dataframe ---")
print(df_fermi)
print("\n--- MAGIC Dataframe ---")
print(df_magic)

# Optional: export to CSV files
df_fermi.to_csv('fermi_lat_data.csv', index=False)
df_magic.to_csv('magic_data.csv', index=False)
