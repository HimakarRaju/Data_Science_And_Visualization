# Get the user input file name
# use libraries for stat analysis -> scipy and specifically use stats
# what we want to do?
"""
a)  Analyze the data set
b) Perform all the stat methods we learned today
c) plot the code graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Analyze function => pass file name as parameter and figure out
# what kind of data it is and what kind of analysis we can do on it
# What kind of fie is given to us i.e., CSV/XLSX


def analyze_dataset(file_path):
    # Read file
    # Extension of file
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise FileNotFoundError
    return data


data = analyze_dataset()

# Descriptive Statistics
print("Descriptive Statistics \n")
print(data.describe())

# Visualize Data
print("Pretty form of the data - Histogram \n")
data.hist(figuresize=(15, 10))
plt.show()

print("Pretty form of the data - Box \n")
data.hist(figuresize=(15, 10))
plt.show()


# Correlation Matrix
print("Correlation Matrix \n")
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.show()

# Hypothesis test - T-Test
print("Hypothesis test (T-Test)")
group1 = data['column1'].values
group2 = data['column2'].values

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f'T-Statistic: {t_stat}, P-value: {p_value} ')

# Regression Analysis
print("Regression Analysis")
X = data[['column1', 'column2']]
Y = data[["column3"]]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, Y_train)
print("Model co-efficients: ", model.coef_)
print("Model Intercept: ", model.intercept_)

# Clustering
print("\n Clustering KMeans : ")
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_
print("Cluster Labels: ", labels)

# Dimensionality Reduction
print("\n Dimensionality Reduction for PCA : ")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)
plt.
plt.show()
