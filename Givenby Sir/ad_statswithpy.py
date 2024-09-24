# Get a user input file name
# use libraries for stat analysis -> scipy and specifically use stats
# What we want to do ?
#   a) Analyze the data set
#   b) Perform all the stat methods we learned today
#   c) Plot the code as graphs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Analyze function - pass filename as parameter and figure out
# what kind of a file is given to us - CSV/XLS

def analyze_dataset(file_path):
    #Read the file and figure out what is it ? How ?
    #Look at the extension of the file.

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("What part of CSV/XLSX files only dont you understand ?")

    # Descriptive Statistics
    print("Descriptive Statistics")
    print(data.describe())

    #Visualise data
    print("PRetty form of data - Histogram \n")
    data.hist(figsize=(15,10))
    plt.show()

    print("PRetty form of data - box \n")
    data.boxplot(figsize=(15,10))
    plt.show()


    #Correlation Matrix
    corr = data[['age','spending']].corr()
    print("Corrleation between age and spending:",corr)
    sns.heatmap(corr,annot=True,cmap='crest')
    plt.show()

    # Hypothesis test - t-test
    print("Hyptothesis test (t-test")
    group1= data['age'].values
    group2= data['purchase_frequency'].values

    t_stat, p_value = stats.ttest_ind(group1,group2)
    print(f"T-Statistic: {t_stat}, P-value: {p_value}")

    #Regression Analysis
    print("Regression Analysis")
    X=data[['age','income']]
    Y=data[['spending']]

    X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
    model=LinearRegression()
    model.fit(X_train,Y_train)
    print("Model co-efficients:",model.coef_)
    print("Model Intercept:",model.intercept_)

    
    #Clustering
    print("\n Clustering (K-means:")
    #data1 = data.select_dtypes(include='number')
    kmeans=KMeans(n_clusters=3)
    kmeans.fit(data[['age','spending']])
    labels=kmeans.labels_
    print("Cluster Labels:",labels)
    
    #Dimensionality Reduction
    print("\nDimensionality reduction for PCA:")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[['age','purchase_frequency']])
    plt.scatter(pca_result[:,0],pca_result[:,1])
    plt.show()


if __name__== "__main__":
    file_path= input("Enter the file name you want me to look at (CSV/Excel ONLY):")
    analyze_dataset(file_path)