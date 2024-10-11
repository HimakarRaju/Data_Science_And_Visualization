import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def suggest_stats(df):
    """Suggest statistical analysis methods and corresponding graph types."""
    stats_suggestions = {}
    
    # Basic statistics
    stats_suggestions['Basic Statistics'] = {
        'Description': 'Calculate mean, median, mode, and standard deviation.',
        'Best Graph': 'Bar Chart or Box Plot'
    }
    
    # Correlation
    stats_suggestions['Correlation Analysis'] = {
        'Description': 'Analyze relationships between features using Pearson/Spearman correlation.',
        'Best Graph': 'Heatmap or Scatter Plot'
    }
    
    # Regression
    stats_suggestions['Linear Regression'] = {
        'Description': 'Predict a continuous target variable based on features.',
        'Best Graph': '3D Scatter Plot for 3D regression'
    }
    
    # Classification
    stats_suggestions['Logistic Regression'] = {
        'Description': 'Predict binary outcomes.',
        'Best Graph': '2D Contour Plot or ROC Curve'
    }
    
    # Clustering
    stats_suggestions['Clustering'] = {
        'Description': 'Group similar data points together.',
        'Best Graph': '3D Scatter Plot or Silhouette Plot'
    }

    # Display suggestions
    for stat, details in stats_suggestions.items():
        print(f"{stat}:\n  {details['Description']}\n  Best Graph: {details['Best Graph']}\n")

def analyze_data(df):
    # Assuming the last column is the target variable
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target

    # Check and convert target variable if necessary
    if y.dtype == 'object':
        print(f"Target column '{target_column}' contains non-numeric values. Converting to numeric.")
        y = y.map({'Yes': 1, 'No': 0})  # Adjust mapping based on your data

        if y.isnull().any():  # Check if mapping resulted in NaN values
            print(f"Warning: Target column '{target_column}' has NaN values after conversion. Cleaning target column.")
            y = y.fillna(y.mean())  # Fill NaN in the target column with the mean or another strategy

    # Ensure no NaN values in the target column
    if y.isnull().any():
        print(f"Target column '{target_column}' contains NaN values after conversion. Cleaning target column.")
        y = y.fillna(y.mean())

    # Separating numeric and non-numeric columns
    X_numeric = X.select_dtypes(include=[float, int])
    X_non_numeric = X.select_dtypes(exclude=[float, int])

    # Print initial row counts
    print(f"Initial number of rows in X_numeric: {len(X_numeric)}")
    
    # Handle missing values for numeric columns
    if X_numeric.isnull().any().any():
        print(f"Numeric feature columns contain NaN values. Cleaning numeric feature columns.")
        X_numeric = X_numeric.fillna(X_numeric.mean())

    # Handle non-numeric columns (optional)
    if not X_non_numeric.empty:
        print("\nNon-numeric data detected. You can either encode or exclude these columns.")
        print(f"Non-numeric columns:\n{X_non_numeric.head()}")

    # Re-check for any NaN values after cleaning
    if X_numeric.isnull().any().any():
        print("Remaining NaN values in numeric columns after cleaning:")
        print(X_numeric.isnull().sum())  # Debugging: print NaN count per column

        # Optionally drop rows with NaN values
        X_numeric = X_numeric.dropna()
        y = y.loc[X_numeric.index]  # Align target variable with cleaned data

    # Print the number of rows after cleaning
    print(f"Number of rows in X_numeric after cleaning: {len(X_numeric)}")
    
    # Check if there are any rows left
    if len(X_numeric) == 0:
        raise ValueError("No data available after cleaning. Please check your dataset and cleaning process.")

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # Check for NaN values in the split data
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        print("NaN values found after train-test split:")
        print(f"X_train NaN counts:\n{X_train.isnull().sum()}")
        print(f"X_test NaN counts:\n{X_test.isnull().sum()}")
        raise ValueError("There are still NaN values in the training or testing data after cleaning.")

    # Perform Linear Regression
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, predictions)
    print(f"Linear Regression R^2 score: {r2}")

    # Plotting the results (Actual vs Predicted)
    plt.scatter(y_test, predictions, label='Predictions vs Actual', color='b')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', lw=2)  # Ideal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.show()

    # 3D Plot for Linear Regression
    if X_numeric.shape[1] >= 2:  # Ensure there are at least two features for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, c='blue', marker='o', label='Actual')
        ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], predictions, c='red', marker='^', label='Predicted')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel(target_column)
        plt.title('3D Scatter Plot for Linear Regression')
        plt.legend()
        plt.show()

    # Standardizing the data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Perform PCA for dimensionality reduction if we have more than two features
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Plot the PCA result
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue')
        plt.title('PCA Result (First 2 Principal Components)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    # 3D Scatter Plot for PCA (if applicable: only if we have more than 2 features)
    if X_numeric.shape[1] >= 3:  # Only plot if we have at least three features
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c='blue')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('3D PCA Result')
        plt.show()

    # Correlation heatmap for numeric data
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_scaled, columns=X_numeric.columns).corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap of Numeric Data')
    plt.show()

def main(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Suggest statistical analysis methods
    suggest_stats(df)

    # Analyze data
    analyze_data(df)

if __name__ == "__main__":
    file_path = "Features data set.csv"  # Update with your CSV file path
    main(file_path)
