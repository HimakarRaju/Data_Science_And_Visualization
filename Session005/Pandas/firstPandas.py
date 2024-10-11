# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:50:32 2024

@author: HimakarRaju
"""

import pandas as pd
# import matplotlib.pyplot as plt

# Read the Excel file
file = 'BirdStrikeDataSet.xlsx'
df = pd.read_excel(file)

# Convert all values to numeric, coercing errors into NaN
df = df.apply(pd.to_numeric, errors='coerce')

print(len(df))


print(df.head(50))

# Calculate column-wise averages
col_averages = df.mean()


col_averages = col_averages.dropna()
print(col_averages)

df.plot.scatter(x=df['Airport: Name'].head(50), y=df['Wildlife: Number Struck Actual'].head(50).mean())
