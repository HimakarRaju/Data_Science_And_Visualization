import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Excel file
file_path = 'bsf.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Ensure relevant columns are numeric
df['Cost: Total $'] = pd.to_numeric(df['Cost: Total $'], errors='coerce')

# Drop rows with NaN values in critical columns
df_cleaned = df.dropna(
    subset=['Cost: Total $', 'Wildlife: Species', 'Aircraft: Make/Model'])

# Group by 'Wildlife: Species' and aggregate by:
# 1. The count of unique 'Aircraft: Make/Model'
# 2. The sum of 'Cost: Total $'
species_damage = df_cleaned.groupby('Wildlife: Species').agg({
    # Count unique aircraft types affected
    'Aircraft: Make/Model': pd.Series.nunique,
    'Cost: Total $': 'sum'  # Sum the total cost
}).reset_index()

# Sort by total cost in descending order and take the top 10 species
top_10_species = species_damage.sort_values(
    by='Cost: Total $', ascending=False).head(10)

# Plot the top 10 bird species by the number of aircraft types affected
plt.figure(figsize=(10, 6))
plt.bar(top_10_species['Wildlife: Species'], top_10_species['Aircraft: Make/Model'],
        color='orange', label='Aircraft Types Affected')
plt.xlabel('Bird Species')
plt.ylabel('Number of Aircraft Types Affected')
plt.title('Top 10 Bird Species by Number of Aircraft Types Affected')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot the top 10 bird species by total cost associated with aircraft damage
plt.figure(figsize=(10, 6))
plt.bar(top_10_species['Wildlife: Species'],
        top_10_species['Cost: Total $'], color='skyblue', label='Total Cost')
plt.xlabel('Bird Species')
plt.ylabel('Total Cost ($)')
plt.title('Top 10 Bird Species by Total Cost of Aircraft Damage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
