import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Excel file
file_path = 'bsf.xlsx'
df = pd.read_excel(file_path)

# Convert 'Cost: Total $' to numeric
df['Cost: Total $'] = pd.to_numeric(df['Cost: Total $'], errors='coerce')

# Drop rows where 'Cost: Total $' is NaN
df_cleaned = df.dropna(subset=['Cost: Total $'])

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_cleaned['Aircraft: Make/Model'],
        df_cleaned['Cost: Total $'], color='skyblue')

# Add labels and title
plt.xlabel('Aircraft: Make/Model')
plt.ylabel('Cost: Total $')
plt.title('Cost by Aircraft: Make/Model')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Display the bar chart
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
