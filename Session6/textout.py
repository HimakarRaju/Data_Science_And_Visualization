import pandas as pd

# Load dataset from Excel file
file_path = 'bsf.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Ensure relevant columns are numeric
df['Cost: Total $'] = pd.to_numeric(df['Cost: Total $'], errors='coerce')

# Drop rows with NaN values in critical columns
df_cleaned = df.dropna(subset=['Cost: Total $', 'Wildlife: Species'])

# Group by 'Wildlife: Species' and aggregate by total cost
species_cost = df_cleaned.groupby('Wildlife: Species').agg({
    'Cost: Total $': 'sum'
}).reset_index()

# Sort by total cost in descending order
species_cost_sorted = species_cost.sort_values(
    by='Cost: Total $', ascending=False)

# Find the bird species that caused the most damage
most_damaging_species = species_cost_sorted.iloc[0]

# Print the result
print(f"The bird species that caused the most damage is '{
      most_damaging_species['Wildlife: Species']}' with a total cost of ${most_damaging_species['Cost: Total $']:.2f}.")
