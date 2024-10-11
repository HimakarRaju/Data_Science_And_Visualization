"""
Create a Python Program that will:
   	a) Automatically Analyze Given Data Set and Suggest Statistical Mechanism to Analyze Data
    b) Decide what graphs are more suitable for visualizing the analysis results 
    c) This Program has to take the Data Set as a command line argument
        
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os


def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
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

        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
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
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=df)
            plt.title(f"Count Plot of {col}")
            plt.show()

    # x= np.arrange(len[])
    # z = np.polyfit(x,values,1)
    # p = np.poly1d(z)

    for col in numerical_cols:
        # plt.figure(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        trendline = go.scatter(x=df[col], y=p(
            go.figure(data=[bar_chart, trendline]())
            fig.update_layout(x='title', y='title')
            x), mode='lines', name='trendline')
        sns.histplot(df[col], kde=False)
        plt.title(f"Distribution Plot of {col}")
        plt.show()

    # for cat_col in categorical_cols:
    #     for num_col in numerical_cols:
    #         if df[cat_col].nunique() <= 10:
    #             plt.figure(figsize=(10, 6))
    #             sns.stripplot(x=cat_col, y=num_col, data=df)
    #             plt.title(f"Strip Plot of {cat_col} vs {num_col}")
    #             plt.show()

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.scatter(df, x=cat_col, y=num_col, color=cat_col, size=num_col,
                                 hover_data=[cat_col, num_col],
                                 title=f"Scatter Plot of {cat_col} vs {num_col}")
                fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate graphs from a data file.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    df = read_data(args.file_path)
    categorical_cols, numerical_cols = analyze_data(df)
    suggestions = suggest_statistical_analysis(df)
    visualize_data(df, categorical_cols, numerical_cols)

    print("Statistical Analysis Suggestions:")
    for col, suggestion in suggestions.items():
        print(f"{col}: {suggestion}")


if __name__ == "__main__":
    main()
