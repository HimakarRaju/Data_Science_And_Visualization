import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Excel file
file = 'bsf.xlsx'
data = pd.read_excel(file)

# Ensure relevant columns are numeric
data['Cost: Total $'] = pd.to_numeric(data['Cost: Total $'], errors='coerce')

# Drop rows with NaN values in critical columns for plotting
data_cleaned = data.dropna(subset=['Aircraft: Make/Model', 'Cost: Total $'])

# Group by 'Aircraft: Make/Model' and total cost
summary = data_cleaned.groupby(
    'Aircraft: Make/Model').agg({'Cost: Total $': 'sum'}).reset_index()

# Sort by cost and take the top 10
top_10_aircraft = summary.sort_values(
    by='Cost: Total $', ascending=False).head(10)

# Plot the top 10 aircrafts by bird strikes
plt.figure(figsize=(10, 6))
plt.bar(top_10_aircraft['Aircraft: Make/Model'],
        top_10_aircraft['Cost: Total $'], color='orange', label='Cost By Aircraft/ Make Model')

# Add labels and title
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Cost: Total $')
plt.title('Top 10 Aircrafts by cost')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
