import argparse
from datetime import time
import pandas as pd
import plotly.express as px
import os

# Step 1 Clubbing all codes with useful things
# Step 2 Get the output from suggest_statistical_analysis
# Step 3 create new data using the results from step 2
# Step 4 plot graphs


def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # # Convert columns to datetime format if they can be parsed as datetime
    # for col in df.columns:
    #     try:
    #         df[col] = pd.to_datetime(
    #             # its generating depreciation for error = "" , used next code.
    #             df[col], errors='ignore', format='%Y-%m-%d %H:%M:%S')
    #     except ValueError:
    #         pass

# # Convert columns to datetime format if they can be parsed as datetime
# for col in df.columns:
#     try:
#         df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')  # it is not skipping any textual columns and raising error
#     except pd.errors.ParserError:
#         # If the column cannot be parsed as datetime, skip it
#         pass

    return df


def analyze_data(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def suggest_statistical_analysis(df):
    suggestions = {}
    for column in df.columns:
        col_data = df[column]
        col_name = column
        unique_values = col_data.nunique()
        dtype = col_data.dtype

        if pd.api.types.is_numeric_dtype(col_data):
            if unique_values == 2:
                suggestions[col_name] = "Binary Logistic Regression"
            elif unique_values < 10:
                suggestions[col_name] = "Chi-square test for independence"
            else:
                suggestions[col_name] = "Linear Regression or ANOVA"

        elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col_data):
            if unique_values <= 10:
                suggestions[col_name] = "Categorical Analysis (e.g., Chi-square, Frequency Distribution)"
            else:
                suggestions[col_name] = "Factor Analysis, Cluster Analysis"

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            suggestions[col_name] = "Time Series Analysis or Trend Analysis"

        elif pd.api.types.is_bool_dtype(col_data):
            suggestions[col_name] = "Binary Logistic Regression"

        else:
            suggestions[col_name] = "Descriptive Statistics (e.g., Mean, Median, Mode)"

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
# # trendline="lowess",  # locally weighted scatterplot smoothing


"""
    # Box Plot
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.box(df, x=cat_col, y=num_col,
                             title=f"Box Plot of {cat_col} vs {num_col}")
                fig.show()

    # violin
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.violin(df, x=cat_col, y=num_col,
                                title=f"Violin Plot of {cat_col} vs {num_col}")
                fig.show()

    # scatter plot
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.scatter(df, x=cat_col, y=num_col, color=cat_col, size=num_col,
                                 hover_data=[cat_col, num_col],
                                 title=f"Scatter Plot of {cat_col} vs {num_col}")
                fig.show()
"""


def graph_suggester(df, categorical_cols, numerical_cols):
    suggestions = {}

    for col in categorical_cols:
        unique_values = df[col].nunique()
        if unique_values <= 10:
            suggestions[col] = ['Bar plot', 'Count Plot', 'Pie chart']
        # else:
        #     suggestions[col] = ['Word cloud', 'Heatmap']

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


# def plot_graphs(df, graphs):
#     for col, graph_types in graphs.items():
#         for graph_type in graph_types:
#             if graph_type == 'Bar plot':
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     fig = px.bar(df, x=df.index, y=col)
#                 else:
#                     fig = px.bar(df, x=col)
#                 fig.show()
#             elif graph_type == 'Count Plot':
#                 fig = px.countplot(df, x=col)
#                 fig.show()
#             elif graph_type == 'Pie chart':
#                 fig = px.pie(df, names=col)
#                 fig.show()
#             elif graph_type == 'Heatmap':
#                 fig = px.density_heatmap(df, x=df.index, y=col)
#                 fig.show()
#             elif graph_type == 'Histogram':
#                 fig = px.histogram(df, x=col)
#                 fig.show()
#             elif graph_type == 'Boxplot':
#                 fig = px.box(df, y=col)
#                 fig.show()
#             elif graph_type == 'Scatterplot':
#                 fig = px.scatter(df, x=df.index, y=col)
#                 fig.show()
#             elif graph_type == 'Time series plot':
#                 fig = px.line(df, x=df.index, y=col)
#                 fig.show()
#             elif graph_type == 'Line chart':
#                 fig = px.line(df, x=df.index, y=col)
#                 fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate graphs from a data file.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    df = read_data(args.file_path)
    categorical_cols, numerical_cols = analyze_data(df)

    graphs = graph_suggester(df, categorical_cols, numerical_cols)
    suggestions = suggest_statistical_analysis(df)
    visualize_data(df, categorical_cols, numerical_cols)

    print("Statistical Analysis Suggestions: \n")
    for col, suggestion in suggestions.items():
        print(f"{col}: {suggestion}")

    print("\nSuggested Graphs : ")
    for col, graph in graphs.items():
        print(f'{col} : {graph}')

    # plot_graphs(df, graphs)


if __name__ == "__main__":
    main()
