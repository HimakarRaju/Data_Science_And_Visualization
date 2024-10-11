import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Function to load data
def load_data(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    return df

# Function to suggest statistical techniques based on data type
def suggest_statistical_analysis(df):
    analysis_suggestions = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            analysis_suggestions[col] = 'Regression, Correlation, Trend Analysis'
        else:
            analysis_suggestions[col] = 'Chi-square test, ANOVA'
    return analysis_suggestions

# Perform correlation and regression analysis
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

    # 3D Plot (if applicable: only if there are exactly 2 features)
    if X_numeric.shape[1] == 2:  # Only plot 3D scatter if we have two features
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, c='blue', marker='o')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel(target_column)
        plt.title('3D Scatter Plot')
        plt.show()

    # Correlation heatmap for numeric data
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_scaled, columns=X_numeric.columns).corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap of Numeric Data')
    plt.show()




# Function to detect trends
def detect_trends(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col].plot(title=f'Trend in {col}')
        plt.show()

# Function to predict trends
def predict_trends(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].values
        model = LinearRegression().fit(X, y)
        trend = model.predict(X)
        plt.plot(df.index, y, label='Actual')
        plt.plot(df.index, trend, label='Predicted', linestyle='--')
        plt.title(f'Trend Prediction in {col}')
        plt.legend()
        plt.show()

# Function to clean data
def clean_data(df):
    # Fill NaN values only for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Optionally, you can handle non-numeric columns separately (e.g., filling with mode or a placeholder)
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')  # Filling with 'Unknown' as a placeholder

    print(f"Cleaned Data:\n{df.head()}")
    return df
def main(file_path):
    # Step 1: Load Data
    df = load_data(file_path)
    
    # Step 2: Clean Data
    df = clean_data(df)
    
    # Step 3: Analyze column relations and suggest techniques
    suggestions = suggest_statistical_analysis(df)
    print("\n--- Suggested Statistical Analysis ---")
    for col, suggestion in suggestions.items():
        print(f"{col}: {suggestion}")

    # Step 4: Perform Analysis
    analyze_data(df)

    # Step 5: Detect Trends
    print("\n--- Trend Detection ---")
    detect_trends(df)

    # Step 6: Predict Trends
    print("\n--- Trend Prediction ---")
    predict_trends(df)

if __name__ == "__main__":
    # Provide the path to your CSV or Excel file
    file_path = 'ssns2.csv'  # Update with your file path
    main(file_path)
