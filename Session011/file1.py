import math
import pandas as pd


file_path = r"C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\session11\customer_data.csv"
data = pd.read_csv(file_path)

num_data = data.select_dtypes(include='number')

list1 = []

for col in num_data:
    list1.append(num_data[col].mean())

print(list1)
