
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file = './customer_transactions.csv'

df = pd.read_csv(file)
data = df


# Performed 4 aggregations grouped on column: 'Account_Type'
df_grp_Account_Type = df.groupby(['Account_Type']).agg(Account_Type_count=('Account_Type', 'count'), Transaction_Amount_sum=('Transaction_Amount', 'sum'), Transaction_Amount_mode=(
    'Transaction_Amount', lambda s: s.value_counts().index[0]), Transaction_Amount_mean=('Transaction_Amount', 'mean')).reset_index()


plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot the count of account types
plt.plot(account_types, account_type_count,
         marker='o', label='Count', color='b')

# Plot the sum of transaction amounts
plt.plot(account_types, transaction_amount_sum,
         marker='s', label='Sum', color='r')

# Customize the plot (add title, labels, legend, etc.)
plt.title('Account Type Analysis')
plt.xlabel('Account Types')
plt.ylabel('Count / Sum of Transaction Amounts')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
