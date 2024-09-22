import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file):
    """Load data from CSV or Excel file."""
    if file.endswith('.csv'):
        return pd.read_csv(file)
    elif file.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise ValueError("File format not supported. Use .csv or .xlsx")


def split_columns(df):
    """Split columns into numeric and non-numeric."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, non_numeric_cols


def plot_graph(df, numeric_col, non_numeric_col):
    """Plot a graph based on the selected numeric and non-numeric columns."""
    plt.figure(figsize=(10, 6))

    # Create a line plot if both selected columns are valid
    if df[non_numeric_col].nunique() <= 10:  # Limit for categorical data
        sns.countplot(data=df, x=non_numeric_col)
        plt.title(f'Count of {non_numeric_col}')
        plt.xlabel(non_numeric_col)
        plt.ylabel('Count')
    else:
        sns.lineplot(data=df, x=non_numeric_col, y=numeric_col, marker='o')
        plt.title(f'{numeric_col} vs {non_numeric_col}')
        plt.xlabel(non_numeric_col)
        plt.ylabel(numeric_col)

    plt.xticks(rotation=0)
    plt.show()


def main():
    # Load data
    file_path = input("Enter the file path (CSV or Excel): ")
    df = load_data(file_path)

    # Split columns
    numeric_cols, non_numeric_cols = split_columns(df)

    print("Numeric columns:", numeric_cols)
    print("Non-numeric columns:", non_numeric_cols)

    # Select columns for plotting
    numeric_col = input(f"Select a numeric column from {numeric_cols}: ")
    non_numeric_col = input(
        f"Select a non-numeric column from {non_numeric_cols}: ")

    # Validate selection
    if numeric_col not in numeric_cols or non_numeric_col not in non_numeric_cols:
        print("Invalid column selection. Please restart the program.")
        return

    # Plot the graph
    plot_graph(df, numeric_col, non_numeric_col)


if __name__ == "__main__":
    main()
