import pandas as pd
import plotly.express as px

import pandas as pd

file = r'Session8\customer_data.xlsx'

df = pd.read_excel(file)
print(df.columns.ravel())


# relation between Income vs. Spending
fig = px.scatter(df, x='Income', y='Spending', title='Income vs. Spending')
fig.show()


# how Purchase_Frequency varies with age
fig = px.bar(df, x='Age', y='Purchase_Frequency',
             title='Age vs. Purchase_Frequency')
fig.show()

# how spending differs between male and female
fig = px.bar(df, x='Gender', y='Spending', title='Average Spending by Gender')
fig.show()

# Mean Income and Spending by Education and Gender
df_clean = df.groupby(['Gender', 'Education']).agg(Income_sum=(
    'Income', 'sum'), Spending_sum=('Spending', 'sum')).reset_index()
print(df_clean['income'])

fig = px.bar(df_clean, x="Education", y=["Income_sum", "Spending_sum"],
             color="Gender", barmode="group",
             category_orders={"Education": df_clean["Education"].unique()},
             labels={"Education": "Level of Education",
                     "Income_sum": "Mean Income",
                     "Spending_sum": "Mean Spending"},
             title="Mean Income and Spending by Education and Gender")
fig.show()
