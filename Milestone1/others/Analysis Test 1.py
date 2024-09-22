import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = '../Python/Data_Science_And_Visualization/Milestone1/customer_transactions.csv'
df = pd.read_csv(file)

# 1. Descriptive Statistics
descriptive = df.describe().reset_index()
print(descriptive)
descriptive.plot(kind="barh", color=plt.cm.tab20(range(len(descriptive))))
plt.xlabel('Values')
plt.ylabel('Statistics')
plt.title('Descriptive Statistics')

# Add labels by index values
for i, v in enumerate(descriptive['index']):
    plt.text(v, i, str(v), color='black', ha='center', va='center')

plt.show()

# 2. Total Transaction Amount by Customer ID
df_Trans_Amt_Per_ID = df.groupby('Customer_ID')[
    'Transaction_Amount'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(df_Trans_Amt_Per_ID['Customer_ID'],
         df_Trans_Amt_Per_ID['Transaction_Amount'], marker='o')
plt.xlabel('Customer ID')
plt.ylabel('Total Transaction Amount')
plt.title('Total Transaction Amount by Customer ID')

plt.show()

# 3. Total Transaction Amount by City
city_totals = df.groupby(
    'City')['Transaction_Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(city_totals.index, city_totals.values,
        color=plt.cm.tab20(range(len(city_totals))))
plt.xlabel('City')
plt.ylabel('Total Transaction Amount')
plt.title('Total Transaction Amount by City')

for i, v in enumerate(city_totals.values):
    plt.text(i, v, str(v), color='black', ha='center', va='bottom')

plt.show()

# 4. Transaction Type Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Transaction_Type', data=df, palette="husl")
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.title('Transaction Type Distribution')

for p in plt.gca().patches:
    plt.gca().text(p.get_x() + p.get_width() / 2, p.get_height(),
                   str(p.get_height()), ha='center', va='bottom')

plt.show()

# 5. Transaction Amount Distribution by Account Type
plt.figure(figsize=(8, 6))
ax = sns.boxplot(x='Account_Type', y='Transaction_Amount',
                 data=df, palette="husl")
plt.xlabel('Account Type')
plt.ylabel('Transaction Amount')
plt.title('Transaction Amount Distribution by Account Type')

plt.show()

# 6. Transaction Amount Distribution
plt.figure(figsize=(8, 6))
df['Transaction_Amount'].plot(kind='hist', bins=30, color='skyblue')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')

plt.show()

# 7. Transaction Amount Count by Account Type
df_Account_Type = df.groupby('Account_Type')[
    'Transaction_Amount'].count().reset_index()
plt.figure(figsize=(8, 6))
plt.bar(df_Account_Type['Account_Type'], df_Account_Type['Transaction_Amount'],
        color=plt.cm.tab20(range(len(df_Account_Type))))
plt.xlabel('Account Type')
plt.ylabel('Transaction Amount Count')
plt.title('Transaction Amount Count by Account Type')

for i, v in enumerate(df_Account_Type['Transaction_Amount']):
    plt.text(i, v, str(v), color='black', ha='center', va='bottom')

plt.show()

# 8. Correlation Heatmap
corr_df = df.select_dtypes("number")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
