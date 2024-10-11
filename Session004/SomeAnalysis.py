import pandas as pd

fname = 'data.csv'
df = pd.read_csv(fname)

col_averages = df.mean()

print(f"Averages : {col_averages}")
