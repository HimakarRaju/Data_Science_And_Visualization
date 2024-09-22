import pandas as pd
import matplotlib.pyplot as plt


# Load dataset from CSV file
file_path = 'data1.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Calculate averages for each numeric column
column_averages = df.mean(numeric_only=True)

# Display the averages
print("Averages of each numeric column:")
print(column_averages)

# Read the Entire Data set
df = pd.read_excel('bsf.xlsx')
print(df)
print(df.columns.ravel())

print(df.info())
# Read a subset of columns from the dataset
ff = pd.read_excel('bsf.xlsx', usecols=['Record ID', 'Aircraft: Type'])
print(ff)

# Filter column by value = Cleanup operation 1 -> remove values which are not valid
fdata = ff[(ff['Aircraft: Type'] != 'Airplane')]
print(fdata)

# Read a CSV File to show how plots are made

dfp = pd.read_csv('data1.csv')  # Read the entire CSV
dfp.plot()  # Plot all the columns and their values without any logic
plt.show()  # Show the graph

# Change the graph type using the kind keyword
dfp.plot(kind='line', x='Duration', y='Maxpulse')
plt.show()

dfp["Duration"].plot(kind='hist')
dfp.plot


plt.show()
