import argparse
import pandas as pd
import plotly.express as px
import os


def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    # Read the file based on the extension
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Handle missing values
    df = df.dropna()

    # Convert columns to datetime if possible
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except (ValueError, TypeError):
            pass

    return df


def analyze_data(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def suggest_statistical_analysis(df):
    suggestions = {}

    for column in df.columns:
        col_data = df[column]
        unique_values = col_data.nunique()

        if pd.api.types.is_numeric_dtype(col_data):
            if unique_values == 2:
                suggestions[column] = "Binary Logistic Regression"
            elif unique_values < 10:
                suggestions[column] = "Chi-square test for independence"
            else:
                suggestions[column] = "Linear Regression or ANOVA"
        elif pd.api.types.is_object_dtype(col_data):
            if unique_values <= 10:
                suggestions[column] = "Categorical Analysis (e.g., Chi-square, Frequency Distribution)"
            else:
                suggestions[column] = "Factor Analysis, Cluster Analysis"
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            suggestions[column] = "Time Series Analysis or Trend Analysis"
        elif pd.api.types.is_bool_dtype(col_data):
            suggestions[column] = "Binary Logistic Regression"
        else:
            suggestions[column] = "Descriptive Statistics (e.g., Mean, Median, Mode)"

    return suggestions


def graph_suggester(df, categorical_cols, numerical_cols):
    suggestions = {}

    for col in categorical_cols:
        unique_values = df[col].nunique()
        if unique_values <= 10:
            suggestions[col] = ['Bar plot', 'Count Plot', 'Pie chart']

    for col in numerical_cols:
        suggestions[col] = ['Histogram', 'Boxplot', 'Scatterplot']

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            suggestions[col] = ['Time series plot', 'Line chart']

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                suggestions[f'{cat_col} vs {num_col}'] = [
                    'Scatterplot', 'Bar plot']

    return suggestions


def visualize_data(df, categorical_cols, numerical_cols):
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            fig = px.bar(df, x=col, color=col, title=f"Count Plot of {col}")
            fig.layout.template = 'presentation'
            fig.show()

    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribution Plot of {col}")
        fig.layout.template = 'presentation'
        fig.show()

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.scatter(df, x=cat_col, y=num_col, color=cat_col, size=num_col,
                                 hover_data=[cat_col, num_col],
                                 title=f"Scatter Plot of {cat_col} vs {num_col}")
                fig.layout.template = 'presentation'
                fig.show()


def main():
    """Main function to execute the workflow."""
    parser = argparse.ArgumentParser(
        description="Automatically generate graphs from a data file.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    # Read and clean data
    df = read_data(args.file_path)

    # Analyze data to get categorical and numerical columns
    categorical_cols, numerical_cols = analyze_data(df)

    # Get suggestions for statistics and graphs
    statistical_suggestions = suggest_statistical_analysis(df)
    graph_suggestions = graph_suggester(df, categorical_cols, numerical_cols)

    # Visualize data
    visualize_data(df, categorical_cols, numerical_cols)

    # Display suggestions
    print("Statistical Analysis Suggestions: \n")
    for col, suggestion in statistical_suggestions.items():
        print(f"{col}: {suggestion}")

    print("\nSuggested Graphs : ")
    for col, graph in graph_suggestions.items():
        print(f'{col}: {graph}')


if __name__ == "__main__":
    main()
