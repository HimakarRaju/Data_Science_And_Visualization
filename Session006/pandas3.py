import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Excel file
file_path = 'bsf.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Ensure relevant columns are numeric
df['Wildlife: Number Struck Actual'] = pd.to_numeric(
    df['Wildlife: Number Struck Actual'], errors='coerce')
df['Cost: Total $'] = pd.to_numeric(df['Cost: Total $'], errors='coerce')

# Drop rows with NaN values in critical columns for plotting
df_cleaned = df.dropna(
    subset=['Wildlife: Number Struck Actual', 'Cost: Total $', 'Aircraft: Make/Model'])

# Group by 'Aircraft: Make/Model' and aggregate by the total number of bird strikes and total cost
summary = df_cleaned.groupby('Aircraft: Make/Model').agg({
    'Wildlife: Number Struck Actual': 'sum',
    'Cost: Total $': 'sum'
}).reset_index()

# Sort by number of bird strikes and take the top 10
top_10_aircrafts = summary.sort_values(
    by='Wildlife: Number Struck Actual', ascending=False).head(10)

# Plot the top 10 aircrafts by bird strikes
plt.figure(figsize=(10, 6))
plt.bar(top_10_aircrafts['Aircraft: Make/Model'],
        top_10_aircrafts['Wildlife: Number Struck Actual'], color='orange', label='Bird Strikes')

# Add labels and title
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Number of Bird Strikes')
plt.title('Top 10 Aircrafts by Number of Bird Strikes')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the top 10 aircrafts by total cost associated with bird strikes
plt.figure(figsize=(10, 6))
plt.bar(top_10_aircrafts['Aircraft: Make/Model'],
        top_10_aircrafts['Cost: Total $'], color='skyblue', label='Total Cost')
# Add labels and title
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Total Cost ($)')
plt.title('Top 10 Aircrafts by Cost Associated with Bird Strikes')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
