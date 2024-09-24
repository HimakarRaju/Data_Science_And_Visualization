import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('Raw_Data.csv')

# 1. Descriptive Statistics
descriptive = df.describe().reset_index()
print(descriptive)
fig, ax = plt.subplots()
descriptive.plot(kind="bar", ax=ax,
                 color=plt.cm.tab20(range(len(descriptive))))
ax.set_xlabel(descriptive['index'])
ax.set_ylabel('Statistics')
ax.set_title('Descriptive Statistics')
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
sns.countplot(x='Transaction_Type', data=df, hue='Transaction_Type')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')  # Replace 'Count' with 'Frequency'
plt.title('Transaction Type Distribution')
for p in plt.gca().patches:
    plt.gca().text(p.get_x() + p.get_width() / 2, p.get_height(),
                   str(p.get_height()), ha='center', va='bottom')

plt.show()

# 5. Transaction Amount Distribution by Account Type
plt.figure(figsize=(8, 6))
sns.boxplot(x='Account_Type', y='Transaction_Amount',
            data=df, hue='Account_Type')
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


# segment Customer_ID based on the sum of Transaction_Amount values compared with the median of the Transaction_Amount:
# Calculate the median of Transaction_Amount
median_transaction_amount = df['Transaction_Amount'].median()

# Calculate the sum of Transaction_Amount for each Customer_ID
customer_transaction_sums = df.groupby(
    'Customer_ID')['Transaction_Amount'].sum().reset_index()

# Create a new column to segment customers based on the sum of Transaction_Amount
customer_transaction_sums['Segment'] = np.where(
    customer_transaction_sums['Transaction_Amount'] > median_transaction_amount, 'High_Value', 'Low_Value')

# Print the resulting dataframe
print(customer_transaction_sums)
# You can also use pd.qcut to segment the customers into more than two groups based on the sum of Transaction_Amount. For example:
customer_transaction_sums['Segment'] = pd.qcut(
    customer_transaction_sums['Transaction_Amount'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
# This will segment the customers into four groups based on the sum of Transaction_Amount: 'Low', 'Medium', 'High', and 'Very_High'.


# plotting code
# Calculate the median of Transaction_Amount
median_transaction_amount = df['Transaction_Amount'].median()

# Calculate the sum of Transaction_Amount for each Customer_ID
customer_transaction_sums = df.groupby(
    'Customer_ID')['Transaction_Amount'].sum().reset_index()

# Create a new column to segment customers based on the sum of Transaction_Amount
customer_transaction_sums['Segment'] = np.where(
    customer_transaction_sums['Transaction_Amount'] > median_transaction_amount, 'High_Value', 'Low_Value')

# Plot the segments using a bar chart
# Calculate the median of Transaction_Amount
median_transaction_amount = df['Transaction_Amount'].median()

# Calculate the sum of Transaction_Amount for each Customer_ID
customer_transaction_sums = df.groupby(
    'Customer_ID')['Transaction_Amount'].sum().reset_index()

# Create a new column to segment customers based on the sum of Transaction_Amount
customer_transaction_sums['Segment'] = np.where(
    customer_transaction_sums['Transaction_Amount'] > median_transaction_amount, 'High_Value', 'Low_Value')

# Plot the segments using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(customer_transaction_sums['Customer_ID'], customer_transaction_sums['Transaction_Amount'], color=[
        'blue' if seg == 'High_Value' else 'red' for seg in customer_transaction_sums['Segment']])
plt.xlabel('Customer ID')
plt.ylabel('Transaction Amount')
plt.title('Customer Segments by Transaction Amount')
plt.show()
