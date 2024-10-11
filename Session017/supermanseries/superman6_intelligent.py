import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def auto_select_target(df):
    # Split columns into numeric and non-numeric
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    if len(numeric_columns) > 0:
        # If numeric columns exist, choose the one with the highest variance as target for regression
        target_column = df[numeric_columns].std().idxmax()  # idxmax() directly gives the column name
        print(f"Automatically selected numeric target column for regression: {target_column}")
        return target_column
    elif len(categorical_columns) > 0:
        # If only categorical columns exist, choose the one with the fewest unique values for classification
        target_column = df[categorical_columns].nunique().idxmin()  # idxmin() directly gives the column name
        print(f"Automatically selected categorical target column for classification: {target_column}")
        return target_column
    else:
        raise ValueError("No suitable target column found in the dataset.")

def preprocess_data(df, target_column):
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Fill NaN values in the target column with the mean (for regression)
    if y.dtype in ['float64', 'int64']:
        y = y.fillna(y.mean())
    
    # Fill NaN values for numeric columns in X
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    
    # Handle categorical columns by filling NaNs and using one-hot encoding
    categorical_columns = X.select_dtypes(include=['object']).columns
    X[categorical_columns] = X[categorical_columns].fillna('Unknown')  # Fill NaN with 'Unknown'
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # If the target is categorical, encode it as well
    if y.dtype == 'object':
        y = pd.get_dummies(y, drop_first=True)
    
    return X, y

def analyze_data(df):
    target_column = auto_select_target(df)
    
    # Preprocess the data
    X, y = preprocess_data(df, target_column)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression().fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    
    # Plotting the predicted vs actual values (for regression tasks)
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{target_column} Prediction')
    plt.show()

def main(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Analyze the data
    analyze_data(df)

if __name__ == "__main__":
    # Set the file path
    file_path = "user_behavior_dataset.csv"  # Replace with your file path
    
    # Run the main function
    main(file_path)
