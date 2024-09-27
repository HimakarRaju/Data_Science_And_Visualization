import pandas as pd
import numpy as np
import argparse
import os

def suggest_statistical_analysis(df):
    suggestions = {}

    for column in df.columns:
        col_data = df[column]
        col_name = column
        unique_values = col_data.nunique()
        dtype = col_data.dtype

        # Detecting if the column is categorical or numerical
        if pd.api.types.is_numeric_dtype(col_data):
            if unique_values == 2:
                suggestions[col_name] = "Binary Logistic Regression"
            elif unique_values < 10:
                suggestions[col_name] = "Chi-square test for independence"
            else:
                suggestions[col_name] = "Linear Regression or ANOVA"
        
        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            if unique_values <= 10:
                suggestions[col_name] = "Categorical Analysis (e.g., Chi-square, Frequency Distribution)"
            else:
                suggestions[col_name] = "Factor Analysis, Cluster Analysis"

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            suggestions[col_name] = "Time Series Analysis or Trend Analysis"
        
        # Special case: binary classification in categorical columns
        elif pd.api.types.is_bool_dtype(col_data):
            suggestions[col_name] = "Binary Logistic Regression"

        else:
            suggestions[col_name] = "Descriptive Statistics (e.g., Mean, Median, Mode)"

    return suggestions

def load_data(file_path):
    # Detect file extension and load the appropriate format
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please provide a .csv or .xlsx file.")
    
    return df

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Suggest statistical analysis for dataset columns.')
    parser.add_argument('datafile', type=str, help='Path to the dataset file (CSV or XLSX format)')
    args = parser.parse_args()

    # Load dataset from command-line argument
    try:
        df = load_data(args.datafile)
    except FileNotFoundError:
        print(f"Error: File '{args.datafile}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File '{args.datafile}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: File '{args.datafile}' is not in a valid format.")
        return
    except ValueError as ve:
        print(ve)
        return

    # Get suggestions for statistical analysis
    analysis_suggestions = suggest_statistical_analysis(df)

    # Print suggestions
    for column, suggestion in analysis_suggestions.items():
        print(f"Column: {column}")
        print(f"Suggested Analysis: {suggestion}\n")

if __name__ == "__main__":
    main()
