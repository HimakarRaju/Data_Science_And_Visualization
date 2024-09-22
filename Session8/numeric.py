import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


file = './customer_transactions.csv'

df = pd.read_csv(file)

df = df.select_dtypes("number")

print("DataFrame:\n", df)

# 1. Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 2. Mean of Columns
print("\nMean of each column:")
print(df.mean())

# 3. Median of Columns
print("\nMedian of each column:")
print(df.median())

# 4. Mode of Columns
print("\nMode of each column:")
print(df.mode())

# 5. Variance of Columns
print("\nVariance of each column:")
print(df.var())

# 6. Standard Deviation of Columns
print("\nStandard Deviation of each column:")
print(df.std())

# 7. Correlation between Columns
print("\nCorrelation between columns:")
print(df.corr())

# 8. Covariance between Columns
print("\nCovariance between columns:")
print(df.cov())

# 9. Minimum and Maximum Values of Each Column
print("\nMinimum values in each column:")
print(df.min())

print("\nMaximum values in each column:")
print(df.max())

# 10. Count of Non-NA/null values
print("\nCount of non-NA/null values in each column:")
print(df.count())

# 11. Skewness of the Data
print("\nSkewness of each column:")
print(df.skew())

# 12. Kurtosis of the Data
print("\nKurtosis of each column:")
print(df.kurt())

# 13. Rolling Mean (Moving Average) for Window of 3
print("\nRolling mean with window of 3:")
print(df.rolling(window=3).mean())

# 14. Cumulative Sum of Columns
print("\nCumulative sum of each column:")
print(df.cumsum())

# 15. Rank of the Values in Each Column
print("\nRank of each value in the DataFrame:")
print(df.rank())

# 16. Quantile (50% or median by default)
print("\n50% Quantile (Median) of each column:")
print(df.quantile(0.5))

# 17. Z-score Standardization
print("\nZ-score Standardization:")
z_score_df = (df - df.mean()) / df.std()
print(z_score_df)

# 18. Percent Change
print("\nPercent Change between each element in DataFrame:")
print(df.pct_change())

# 19. Correlation Heatmap (using seaborn for visualization)

print("\nCorrelation Heatmap:")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
