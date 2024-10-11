import logging
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, Normalizer  # Import Normalizer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Setup Plotly renderer
pio.renderers.default = "browser"


# Setup logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Check for symbols and spaces in column names
def check_column_names(df):
    invalid_columns = [col for col in df.columns if re.search(r"[^a-zA-Z0-9_]", col)]
    if invalid_columns:
        print(f"Warning: Columns with symbols/spaces detected: {invalid_columns}")
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    return df


# Create directories
def create_directories(output_directory, dataset_name):
    dirs = ["analysis", "plots", "logs", "pkl"]
    return {d: os.path.join(output_directory, dataset_name, d) for d in dirs}


# Load target column data from pickle
def load_target_from_pickle(pkl_dir):
    pkl_file = os.path.join(pkl_dir, "target_column.pkl")
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            target_column = pickle.load(f)
        print(f"Previous target column found: {target_column}")
        return target_column
    return None


# Save target column to pickle
def save_target_to_pickle(target_column, pkl_dir):
    pkl_file = os.path.join(pkl_dir, "target_column.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(target_column, f)


# Suggest target column using NLP or previous data
def suggest_target_column(df, pkl_dir):
    target_column = load_target_from_pickle(pkl_dir)
    if not target_column:
        for column in df.columns:
            if any(x in column.lower() for x in ["price", "target", "label"]):
                target_column = column
                break
        if not target_column:
            target_column = input(
                f"No suitable target column found. Choose from: {list(df.columns)} "
            )

    print(f"Suggested target column: {target_column}")
    confirmation = input("Is this okay? (y/n): ")
    if confirmation.lower() != "y":
        target_column = input(
            f"Please choose the target column from: {list(df.columns)} "
        )

    save_target_to_pickle(target_column, pkl_dir)
    return target_column


# Data transformation
def transform_data(df):
    print("Starting data transformation...")
    imputer = SimpleImputer(strategy="mean")
    df_imputed = df.copy()

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df_imputed[numerical_columns] = imputer.fit_transform(df_imputed[numerical_columns])

    # Normalization
    if df_imputed[numerical_columns].max().max() != 0:
        normalizer = Normalizer()
        df_imputed[numerical_columns] = normalizer.fit_transform(
            df_imputed[numerical_columns]
        )

    # Label encoding for categorical variables
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df_imputed[column] = le.fit_transform(df_imputed[column].astype(str))
        label_encoders[column] = le

    print("Data transformation completed.")
    return df_imputed, label_encoders


# Log errors
def log_error(model_name, error, log_file):
    logging.error(f"Model {model_name} failed with error: {error}")


# Save the best model and its metrics
def save_best_model(results, doc_filename):
    if results:  # Check if results is not empty
        best_model = max(results, key=lambda model: results[model]["r2"])
        print(f"Best Model: {best_model} with R2: {results[best_model]['r2']}")
        with open(doc_filename, "a") as f:
            f.write(f"Best Model: {best_model} - Metrics: {results[best_model]}\n")
    else:
        print("No models were successfully trained.")


# Generate and save plots
def generate_and_save_plots(df, plots_dir):
    print("Generating plots...")
    # Histograms
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        fig.write_html(os.path.join(plots_dir, f"histogram_{col}.html"))

    # Correlation matrix
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    fig.write_html(os.path.join(plots_dir, "correlation_matrix.html"))


# Main analysis function
def run_analysis(file_path, output_directory):
    try:
        df = load_file(file_path)
        if df.empty:
            print("No data to process.")
            return

        # Create directories
        dataset_name = os.path.basename(file_path).split(".")[0]
        dirs = create_directories(output_directory, dataset_name)
        for dir in dirs.values():
            os.makedirs(dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(dirs["logs"], f"{dataset_name}_error_log.log")
        setup_logging(log_file)

        # Check and fix column names
        df = check_column_names(df)

        # Suggest the target column
        target_column = suggest_target_column(df, dirs["pkl"])

        # Transform Data
        df_imputed, label_encoders = transform_data(df)

        # Split the data
        X = df_imputed.drop(columns=[target_column])
        y = df_imputed[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
        }

        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = {"mse": mse, "r2": r2}
                print(f"{model_name} - MSE: {mse:.2f}, R2: {r2:.2f}")
            except Exception as e:
                log_error(model_name, e, log_file)

        # Generate and save plots
        generate_and_save_plots(df_imputed, dirs["plots"])

        # Save the best model and metrics
        save_best_model(results, os.path.join(dirs["analysis"], "model_report.txt"))

    except Exception as e:
        logging.error(f"Main analysis failed with error: {e}")


def load_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return pd.DataFrame()


if __name__ == "__main__":
    run_analysis(
        "C:\\Users\\HimakarRaju\\Desktop\\Milestone2\\Datasets\\laptop_prices.csv",
        "C:\\Users\\HimakarRaju\\Desktop\\Milestone2\\DataReadOuts1",
    )
