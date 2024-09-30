import pandas as pd


file = r'C:\\Users\\HimakarRaju\\Desktop\\PreSkilling\\Python\\Data_Science_And_Visualization\\A_A_Explanation\\SPF.csv'

# reading the file
df = pd.read_csv(file)

print(df.describe())


# checking data
print(df.head())

# checking null values
print(df.isnull().sum())

# deleting rows with null values
df.dropna(inplace=True)
print(df.isnull().sum())

# separating numerical and categorical columns
categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

print(categorical_cols)
print(numerical_cols)
