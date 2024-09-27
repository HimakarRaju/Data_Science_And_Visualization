import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def read_data(file_path):
    """Read data from CSV or Excel file and return a DataFrame."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return df

def detect_field_types(df):
    """Detect field types in the DataFrame and return numeric and categorical columns."""
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_columns, categorical_columns

def generate_graphs(df):
    """Generate graphs for numeric and categorical columns."""
    numeric_columns, categorical_columns = detect_field_types(df)

    # Generate histograms for numeric columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, color='blue')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Generate bar plots for categorical columns
    for col in categorical_columns:
        if df[col].nunique() < 20:  # Only plot bar charts for columns with fewer than 20 unique values
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col], data=df, palette='Set2')
            plt.title(f'Bar Plot of {col}')
            plt.ylabel(col)
            plt.xlabel('Count')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def main():
    # Use argparse to parse the file path argument
    parser = argparse.ArgumentParser(description="Automatically generate graphs from a data file.")
    parser.add_argument('file_path', type=str, help="Path to the input CSV or Excel file.")
    
    # Parse the command-line argument
    args = parser.parse_args()
    
    # Read data from the input file
    df = read_data(args.file_path)
    
    # Generate graphs based on the data
    generate_graphs(df)

if __name__ == '__main__':
    main()
