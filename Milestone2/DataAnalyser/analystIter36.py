import logging
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Setup Plotly renderer
pio.renderers.default = "browser"


# Setup logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Function to check for symbols and spaces in column names
def check_column_names(df):
    invalid_columns = [col for col in df.columns if re.search(r"[^a-zA-Z0-9_]", col)]
    if invalid_columns:
        print(f"Warning: Columns with symbols/spaces detected: {invalid_columns}")
    return invalid_columns


# Function to create directories
def create_directories(output_directory, dataset_name):
    analysis_dir = os.path.join(output_directory, dataset_name, "analysis")
    plots_dir = os.path.join(output_directory, dataset_name, "plots")
    logs_dir = os.path.join(output_directory, dataset_name, "logs")
    pkl_dir = os.path.join(output_directory, dataset_name, "pkl")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    return analysis_dir, plots_dir, logs_dir, pkl_dir


# Function to load pickled target column data
def load_target_from_pickle(pkl_dir):
    pkl_file = os.path.join(pkl_dir, "target_column.pkl")
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            target_column = pickle.load(f)
        print(f"Previous target column found: {target_column}")
        return target_column
    return None


# Function to save target column to pickle
def save_target_to_pickle(target_column, pkl_dir):
    pkl_file = os.path.join(pkl_dir, "target_column.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(target_column, f)


# Function to save all data to pickle
def save_all_data_to_pickle(df, pkl_dir):
    pkl_file = os.path.join(pkl_dir, "full_data.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(df, f)


# Function to suggest target column using NLP or previous data
def suggest_target_column(df, pkl_dir):
    target_column = load_target_from_pickle(pkl_dir)

    if not target_column:
        for column in df.columns:
            column_name = column.lower()
            if (
                "price" in column_name
                or "target" in column_name
                or "label" in column_name
            ):
                target_column = column
                break
        if not target_column:
            target_column = input(
                "No suitable target column found by NLP. Please select one from the following: "
                + str(list(df.columns))
            )

    # Confirm with user
    print(f"Suggested target column: {target_column}")
    confirmation = input("Is this okay? (y/n): ")
    if confirmation.lower() != "y":
        target_column = input(
            "Please choose the target column from the list: " + str(list(df.columns))
        )

    save_target_to_pickle(target_column, pkl_dir)
    return target_column


# Function to perform data transformation
def transform_data(df):
    transformation_report = []
    imputer = SimpleImputer(strategy="mean")
    df_imputed = df.copy()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df_imputed[numerical_columns] = imputer.fit_transform(df_imputed[numerical_columns])
    transformation_report.append(
        "Missing values were imputed using the SimpleImputer with mean strategy."
    )

    normalizer = Normalizer()
    if df_imputed[numerical_columns].max().max() != 0:
        df_imputed[numerical_columns] = normalizer.fit_transform(
            df_imputed[numerical_columns]
        )
        transformation_report.append("Normalization applied to numerical features.")
    else:
        transformation_report.append("Normalization skipped due to zero values.")

    label_encoders = {}
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df_imputed[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    transformation_report.append("Label encoding applied to categorical variables.")

    return df_imputed, label_encoders, transformation_report


# Handle errors during model execution
def log_error(model_name, error, log_file):
    logging.error(f"Model {model_name} failed with error: {error}")


# Function to save the best model and its metrics
def save_best_model(results, doc_filename):
    best_model = max(
        results,
        key=lambda model: (
            results[model]["accuracy"]
            if "accuracy" in results[model]
            else results[model]["r2"]
        ),
    )
    print(f"Best Model: {best_model}")
    print(f"Best Model Metrics: {results[best_model]}")

    with open(doc_filename, "a") as f:
        f.write(f"\nBest Model: {best_model}\n")
        f.write(f"Best Model Metrics: {results[best_model]}\n")


# Function to generate and save plots using Plotly
def generate_and_save_plots(df, plots_dir):
    # Histogram with KDE lines for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, marginal="box", title=f"Histogram of {col}")
        fig.update_traces(opacity=0.75)
        fig.write_html(os.path.join(plots_dir, f"histogram_{col}.html"))

    # Scatter plots between target column and numerical features
    target_col = df.select_dtypes(include=[np.number]).columns[
        -1
    ]  # Assume the last numeric column is the target
    for col in numerical_cols:
        if col != target_col:
            fig = px.scatter(
                df, x=col, y=target_col, title=f"Scatter plot of {col} vs {target_col}"
            )
            fig.write_html(
                os.path.join(plots_dir, f"scatter_{col}_vs_{target_col}.html")
            )

    # Correlation matrix plot
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    fig.write_html(os.path.join(plots_dir, "correlation_matrix.html"))

    # Time series analysis
    date_columns = df.select_dtypes(include=["datetime", "object"]).columns
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
            if df[col].isnull().sum() == 0:
                df.set_index(col, inplace=True)
                fig = px.line(df, title=f"Time Series Analysis for {col}")
                fig.write_html(os.path.join(plots_dir, f"time_series_{col}.html"))
                df.reset_index(inplace=True)
        except Exception as e:
            log_error(col, e, "time_series_error.log")


# Main function to invoke all processes
def run_analysis(file_path, output_directory):
    try:
        df = load_file(file_path)
        if df.empty:
            print("No data to process.")
            return

        # Create directories
        dataset_name = os.path.basename(file_path).split(".")[0]
        analysis_dir, plots_dir, logs_dir, pkl_dir = create_directories(
            output_directory, dataset_name
        )

        # Setup logging
        log_file = os.path.join(logs_dir, f"{dataset_name}_error_log.log")
        setup_logging(log_file)

        # Check column names
        invalid_columns = check_column_names(df)
        if invalid_columns:
            df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_")

        # Suggest the target column
        target_column = suggest_target_column(df, pkl_dir)

        # Save all data to pickle
        save_all_data_to_pickle(df, pkl_dir)

        print("Transforming Data")
        df_imputed, label_encoders, transformation_report = transform_data(df)

        # Split the data
        X = df_imputed.drop(columns=[target_column])
        y = df_imputed[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVC": SVC(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
        }

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[model_name] = {"accuracy": accuracy}
                print(f"{model_name} Accuracy: {accuracy}")

                # For regression models, use different metrics
                if model_name in ["Random Forest", "Gradient Boosting", "KNN", "SVC"]:
                    regression_model = (
                        RandomForestRegressor()
                        if model_name == "Random Forest"
                        else None
                    )
                    regression_model.fit(X_train, y_train)
                    y_pred_reg = regression_model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred_reg)
                    r2 = r2_score(y_test, y_pred_reg)
                    results[model_name].update({"mse": mse, "r2": r2})
                    print(f"{model_name} MSE: {mse}, R2: {r2}")
            except Exception as e:
                log_error(model_name, e, log_file)

        # Generate and save plots
        generate_and_save_plots(df, plots_dir)

        # Save the best model and metrics
        save_best_model(results, os.path.join(analysis_dir, "model_report.txt"))

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
        "C:\\Users\\HimakarRaju\\Desktop\\Milestone2\\DataReadOuts",
    )
