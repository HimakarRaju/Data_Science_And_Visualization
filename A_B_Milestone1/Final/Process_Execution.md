# Milestone 1

## Instructions before the project execution

    1. READ THE ENTIRE DOCUMENT BEFORE BEGINNING
    2. Write down the understanding
    3. PLAN properly start to end the path of working on this project
    4. Do not use any code generation tools for completion
    5. In the project review, questions about any of the above can be asked, including code, logic and the reasoning. 
    6. Hence it is advised to do the project independently.
    7. Document the project execution from start to finish
    8. Document the code, the logic, the reasoning and the challenges faced during the project

## Project Overview

The goal of this project is to give you hands-on experience in applying Python-based data science
techniques using libraries like Pandas, NumPy, Matplotlib, and Seaborn.

With a dataset of 10,000 financial transactions, You will explore various aspects of
a. Data manipulation,
b. Cleaning,
c. Visualization with Python, pandas, matplotlib, numpy and seaborn to achieve the results.

The dataset is a collection of 10,000 financial transactions, with each transaction represented by
a unique identifier, date(time), amount, and category.

### The objective is to analyze the data to answer the following questions

1. What is the total amount spent in each category?
2. The Account type most used
3. The type of transactions most made
4. The average amount spent in each category
5. The total amount spent in each region

### Steps Done

### Transaction Amount by Hour of Day

df_hourly = df.resample('H').mean()
fig = px.line(df_hourly, x=df_hourly.index, y='Transaction_Amount', title='Transaction Amount by Hour of Day')

### Customer Transaction Frequency

customer_freq = df.groupby('Customer_ID').size().reset_index(name='Transaction_Frequency')
fig = px.bar(customer_freq, x='Customer_ID', y='Transaction_Frequency', title='Customer Transaction Frequency')

### City-wise Transaction Amount by Account Type

city_account_type = df.groupby[['City', 'Account_Type']](../'Transaction_Amount').sum().reset_index()
fig = px.bar(city_account_type, x='City', y='Transaction_Amount', color='Account_Type', title='City-wise Transaction Amount by Account Type')

### Transaction Amount by Day of Week

df_daily = df.resample('D').mean()
fig = px.line(df_daily, x=df_daily.index, y='Transaction_Amount', title='Transaction Amount by Day of Week')

### Correlation Analysis by Account Type

corr_by_account_type = {}
for account_type in df['Account_Type'].unique():
    corr_df = df[df['Account_Type'] == account_type].select_dtypes("number")
    corr_by_account_type[account_type] = corr_df.corr()
fig = px.imshow(corr_by_account_type, title="Correlation Analysis by Account Type")

### Graphs to consider

1. Bar and Horizontal Bar Plots: Good for general comparison and when categories (customer IDs) are numerous.
2. Pie Chart: Effective for showing proportions but can become cluttered with many categories.
3. Line Plot: Useful for ordered or time-series data.
4. Scatter Plot: Good for spotting outliers or patterns in individual customer behaviors.
5. Box and Violin Plots: Ideal for understanding the distribution of transaction amounts and identifying outliers.
6. Histogram: Helps understand the frequency of various transaction amounts.
7. Stacked Bar Chart: Useful for comparing multiple variables per customer.

### Some predicting possibilities

1. Customer Spending Patterns:
   1. Analyze transaction amounts by account type to identify trends in spending.  
   2. For instance, you could determine if customers are spending more from their checking accounts versus savings.

2. Account Health Assessment:
   1. Predict the likelihood of customers overdrawing their checking accounts based on their transaction history and patterns in spending.

3. Segmentation of Customers:
   1. Group customers by their transaction behaviors or account types to identify distinct segments (e.g., high spenders, savers, or those relying heavily on credit).

4. Credit Utilization:
   1. Analyze credit usage over time to predict future borrowing behaviors.
   2. You could determine the average credit utilization rate and identify customers at risk of exceeding their limits.

5. Geographic Insights:
   1. Analyze how transaction behaviors vary by city.
   2. This can reveal regional trends or economic conditions affecting spending and saving.

### Observations

   1. **Credit Accounts**: The total transaction amount for credit accounts is $791,139.06.
      1. This indicates that, overall, credit transactions resulted in a positive flow of money, suggesting that users are utilizing their credit lines.

   2. **Savings Accounts**: The total transaction amount for savings accounts is $275,982.56.
      1. This suggests a net positive flow, likely indicating deposits and interest earnings exceeding withdrawals.

   3. **Checking Accounts**: The total transaction amount for checking accounts is -$512,854.47.
      1. The negative value indicates that the withdrawals or expenses from checking accounts exceeded the deposits. This might reflect common behavior where checking accounts are used for day-to-day transactions, resulting in a net negative balance.
