import logging
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, Normalizer


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


# Function to plot and save histograms with KDE
def plot_histograms(df, target_column, plots_dir):
    for column in df.columns:
        fig = px.histogram(
            df,
            x=column,
            marginal="violin",
            nbins=30,
            title=f"{column} Distribution with KDE",
        )
        fig.update_layout(bargap=0.1)
        fig.write_html(os.path.join(plots_dir, f"{column}_histogram_kde.html"))


# Function to plot and save scatter plots between target and other features
def plot_scatter_plots(df, target_column, plots_dir):
    for column in df.columns:
        if column != target_column:
            fig = px.scatter(
                df,
                x=column,
                y=target_column,
                title=f"Scatter Plot: {column} vs {target_column}",
            )
            fig.write_html(
                os.path.join(plots_dir, f"{column}_vs_{target_column}_scatter.html")
            )


# Function to plot and save PCA components
def plot_pca(pca, X_train, plots_dir):
    fig = px.scatter_matrix(
        pd.DataFrame(
            pca.transform(X_train),
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        ),
        title="PCA Scatter Matrix",
    )
    fig.write_html(os.path.join(plots_dir, "pca_scatter_matrix.html"))


# Function to plot and save correlation heatmap
def plot_correlation_heatmap(df, plots_dir):
    corr_matrix = df.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="Viridis",
        )
    )
    fig.update_layout(title="Correlation Heatmap")
    fig.write_html(os.path.join(plots_dir, "correlation_heatmap.html"))


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

        print("Transforming Data")
        df_imputed, label_encoders, transformation_report = transform_data(df)

        # Generate and save plots
        plot_histograms(df_imputed, target_column, plots_dir)
        plot_scatter_plots(df_imputed, target_column, plots_dir)
        plot_correlation_heatmap(df_imputed, plots_dir)

        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(df_imputed)
        plot_pca(pca, df_imputed, plots_dir)

        # Add model training and selection here
        # ...

    except Exception as e:
        log_error("General Pipeline", str(e), log_file)


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
