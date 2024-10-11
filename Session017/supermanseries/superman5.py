import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import sqlite3

def load_data(file_path, source_type='csv', sql_query=None, sql_conn=None):
    """Loads data from various sources."""
    if source_type == 'csv':
        return pd.read_csv(file_path)
    elif source_type == 'excel':
        return pd.read_excel(file_path)
    elif source_type == 'sql':
        if sql_query and sql_conn:
            return pd.read_sql_query(sql_query, sql_conn)
        else:
            raise ValueError("SQL query and connection must be provided for SQL source.")
    else:
        raise ValueError("Unsupported source type.")

def preprocess_data(df, target_column):
    """Preprocess the data: handle missing values, encode categorical variables, and scale numeric data."""
    # Check if target column exists
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset. Available columns are: {df.columns.tolist()}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical variables
    
    # Label encode the target variable if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    return X, y

def suggest_model(y):
    """Suggest a machine learning model based on the target variable type."""
    if y.nunique() == 2:  # Binary classification
        return LogisticRegression()
    elif pd.api.types.is_numeric_dtype(y):  # Regression for numeric target
        return LinearRegression()
    else:
        raise ValueError("The program currently supports binary classification and regression tasks only.")

def suggest_graph(X, y):
    """Suggest the best graph for the data analysis based on the input data."""
    if X.shape[1] == 2:
        return "2D Scatter Plot"
    elif X.shape[1] > 2:
        return "3D Scatter Plot"
    else:
        return "Simple Line Plot"

def analyze_data(X, y, model, graph_type):
    """Train the model, make predictions, and visualize results based on the graph type."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    if isinstance(model, LinearRegression):
        score = r2_score(y_test, predictions)
        print(f"Linear Regression R^2 score: {score}")
    else:
        score = accuracy_score(y_test, (predictions > 0.5).astype(int))  # For Logistic Regression
        print(f"Logistic Regression Accuracy score: {score}")

    # Visualize results based on the suggested graph type
    if graph_type == "3D Scatter Plot" and X.shape[1] >= 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, c='blue', label='Actual')
        ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], predictions, c='red', marker='^', label='Predicted')
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel('Target')
        plt.title('3D Scatter Plot')
        plt.legend()
        plt.show()
    elif graph_type == "2D Scatter Plot" and X.shape[1] == 1:
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, predictions, color='red', label='Predicted')
        plt.xlabel(X.columns[0])
        plt.ylabel('Target')
        plt.title('2D Scatter Plot')
        plt.legend()
        plt.show()

def suggest_stats(df):
    """Suggest statistical methods and appropriate graphs."""
    stats_suggestions = {
        'Basic Statistics': 'Bar Chart or Box Plot',
        'Correlation Analysis': 'Heatmap or Scatter Plot',
        'Regression Analysis': '3D Scatter Plot (if applicable)',
        'Classification': '2D Contour Plot or ROC Curve',
        'Clustering': '3D Scatter Plot or Silhouette Plot'
    }
    
    print("\nSuggested Statistical Methods and Graphs:")
    for method, graph in stats_suggestions.items():
        print(f"{method}: Best Graph - {graph}")

def main(file_path, target_column, source_type='csv', sql_query=None, sql_conn=None):
    # Load the data
    df = load_data(file_path, source_type, sql_query, sql_conn)
    
    # Print the available columns in the dataset to help debug the issue
    print(f"Columns in the dataset: {df.columns.tolist()}")
    
    # Preprocess the data
    X, y = preprocess_data(df, target_column)
    
    # Suggest a model based on the target variable type
    model = suggest_model(y)
    
    # Suggest an appropriate graph
    graph_type = suggest_graph(X, y)
    
    # Analyze the data and plot the results
    analyze_data(X, y, model, graph_type)

if __name__ == "__main__":
    # Example: Change file_path and target_column based on your data
    file_path = "bsf.csv"  # CSV, Excel, or SQL path
    target_column = "target"  # Change this to your actual target column name
    
    # Uncomment and use the appropriate data source
    source_type = 'csv'
    #source_type = 'excel'
    
    # For SQL
    # sql_conn = sqlite3.connect('database.db')
    # sql_query = "SELECT * FROM your_table"
    # source_type = 'sql'
    
    main(file_path, target_column)
