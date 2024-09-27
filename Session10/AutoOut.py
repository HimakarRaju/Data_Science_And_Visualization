# Import necessary libraries
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate graphs from a data file.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    df = read_data(args.file_path)
    process_data(df)


def process_data(df):
    print("Initial Data Overview:")
    print(df.head(10))

    # Clean data
    df = clean_data(df)
    print(df.describe())
    # Format columns
    df, label_encoders = format_columns(df)

    # Show statistics
    stats = get_statistics(df)
    print("Statistics of Numerical Columns:")
    print(stats)

    # Plot graphs
    plot_best_graph(df)


# Step 1: Load the dataset
def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    return df


# Step 2: Clean data (handling missing values and outliers)
def clean_data(df):
    # Fill missing values with median for numerical columns and mode for categorical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df


# Step 3: Format columns (Convert categorical variables, etc.)
def format_columns(df):
    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store the encoders if needed later

    return df, label_encoders


# Step 4: Find best column statistics (mean, median, std for numerical columns)
def get_statistics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    stats = df[num_cols].describe().T
    return stats[['mean', '50%', 'std']]  # '50%' is median


# Step 5: Plot the best graph based on column types
def plot_best_graph(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Plotting categorical vs numerical
    for cat_col in cat_cols:
        for num_col in num_cols:
            unique_vals = df[cat_col].nunique()

            if unique_vals <= 10:  # Scatter plot for small categories
                plt.figure(figsize=(8, 5))
                sns.scatterplot(x=df[cat_col], y=df[num_col])
                plt.title(f'Scatter Plot of {num_col} vs {cat_col}')
                plt.show()
            else:  # Boxplot for larger categories
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=df[cat_col], y=df[num_col])
                plt.title(f'Boxplot of {num_col} by {cat_col}')
                plt.show()

    # Plot a heatmap of correlations for numerical columns (outside of the loop)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()


if __name__ == "__main__":
    main()
