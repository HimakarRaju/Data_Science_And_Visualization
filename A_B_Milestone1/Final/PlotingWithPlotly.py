import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# step 1 reading file
file = 'C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\A_B_Milestone1\Final\customer_transactions.csv'
df = pd.read_csv(file)


# 1. Descriptive Statistics
descriptive = df.describe().reset_index()
fig = px.bar(descriptive, x='index', y='Transaction_Amount',
             title='Descriptive Statistics', color='Transaction_Amount')
print(descriptive)
fig.show()

# 2. Total Transaction Amount by Customer ID
df_Trans_Amt_Per_ID = df.groupby('Customer_ID')[
    'Transaction_Amount'].sum().reset_index()
fig = px.line(df_Trans_Amt_Per_ID, x='Customer_ID', y='Transaction_Amount',
              title='Total Transaction Amount by Customer ID')

fig.show()

# 3. Total Transaction Amount by City
city_totals = df.groupby(
    'City')['Transaction_Amount'].sum().sort_values(ascending=False)
fig = px.bar(city_totals, x=city_totals.index, y=city_totals.values,
             title='Total Transaction Amount by City', color='Transaction_Amount')
fig.show()

# 4. Transaction Type Distribution
fig = px.pie(df, names='Transaction_Type',
             title='Transaction Type Distribution')
fig.show()

# 5. Transaction Amount Distribution by Account Type
fig = px.box(df, x='Account_Type', y='Transaction_Amount',
             title='Transaction Amount Distribution by Account Type')
fig.show()

# 6. Transaction Amount Distribution
fig = go.Figure(data=[go.Histogram(x=df['Transaction_Amount'], nbinsx=30)])
fig.add_vline(x=df['Transaction_Amount'].mean(), line_width=2, line_dash="dash",
              line_color="red", annotation_text="Mean", annotation_position="top right")
fig.add_vline(x=df['Transaction_Amount'].median(), line_width=2, line_dash="dash",
              line_color="green", annotation_text="Median", annotation_position="top right")
fig.update_layout(title_text='Transaction Amount Distribution',
                  xaxis_title_text='Transaction Amount', yaxis_title_text='Frequency')
fig.show()

# 7. Transaction Amount Count by Account Type
df_Account_Type = df.groupby('Account_Type')[
    'Transaction_Amount'].count().reset_index()
fig = px.pie(df_Account_Type, names='Account_Type', values='Transaction_Amount',
             title='Transaction Amount Count by Account Type', color='Account_Type', color_discrete_map={'Checking': 'blue', 'Savings': 'green'})
fig.show()

# 8. Correlation Heatmap
corr_df = df.select_dtypes("number")
fig = px.imshow(corr_df.corr(), title="Correlation Heatmap")
fig.show()


# 10.
# 'Transaction_Date' format is like : 00:00.0 - 59:59.9
"""
Tried converting to time but as the output is linear there was no particular results
    # Convert Transaction_Date to datetime
    df['Transaction_Date'] = pd.to_datetime(
    df['Transaction_Date'], format='%M:%S.%f')
 """


# Trying to generate random dates over last two years
# Generate random dates between 2021 and 2023
start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2023-12-31')
num_random_dates = df.shape[0]  # You can choose how many random dates you want

random_dates = pd.date_range(
    start=start_date, end=end_date, periods=num_random_dates)
df['Random_Transaction_Date'] = random_dates

# fig = px.line(city_date_amount, x='Random_Transaction_Date', y='Transaction_Amount',
#               color='City', barmode='group', title='Transaction Amount in Various Cities Over Time')
# fig.show()

city_date_amount = df.groupby(['City', 'Random_Transaction_Date'])[
    'Transaction_Amount'].sum().reset_index()

# Plot the data
fig = px.line(city_date_amount, x='Random_Transaction_Date', y='Transaction_Amount',
              title='Transaction Amount in Various Cities Over Time', color='City')
fig.show()


# trying prediction plots
# Predict transaction amount by city
city_amount = df.groupby('City')['Transaction_Amount'].mean().reset_index()
fig = px.bar(city_amount, x='City', y='Transaction_Amount',
             title='Average Transaction Amount by City', color='Transaction_Amount')
fig.add_trace(go.Scatter(x=city_amount['City'], y=[city_amount['Transaction_Amount'].mean(
)]*len(city_amount), mode='lines', line=dict(color='red', dash='dash')))
fig.show()

# Predict transaction amount by account type
account_amount = df.groupby('Account_Type')[
    'Transaction_Amount'].mean().reset_index()
fig = px.bar(account_amount, x='Account_Type', y='Transaction_Amount',
             title='Average Transaction Amount by Account Type', color='Transaction_Amount')
fig.add_trace(go.Scatter(x=account_amount['Account_Type'], y=[account_amount['Transaction_Amount'].mean(
)]*len(account_amount), mode='lines', line=dict(color='green', dash='dash')))
fig.show()

# Predict transaction amount by transaction type
transaction_amount = df.groupby('Transaction_Type')[
    'Transaction_Amount'].mean().reset_index()
fig = px.bar(transaction_amount, x='Transaction_Type', y='Transaction_Amount',
             title='Average Transaction Amount by Transaction Type')
fig.add_trace(go.Scatter(x=transaction_amount['Transaction_Type'], y=[transaction_amount['Transaction_Amount'].mean(
)]*len(transaction_amount), mode='lines', line=dict(color='blue', dash='dash')))
fig.show()


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
fig = px.bar(customer_transaction_sums, x='Customer_ID', y='Transaction_Amount',
             color='Segment',  title='Customer Transaction Amount into four Segmentations', range_color=customer_transaction_sums['Segment'])
fig.show()

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
fig = px.bar(customer_transaction_sums, x='Customer_ID', y='Transaction_Amount',
             color='Segment', color_discrete_map={'High_Value': 'blue', 'Low_Value': 'red'})
fig.update_layout(title='Customer Segments by Transaction Amount',
                  xaxis_title='Customer ID', yaxis_title='Transaction Amount')
fig.show()
