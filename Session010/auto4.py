import argparse
import os
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset


def read_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Step 2: Clean data (handling missing values and outliers)


def clean_data(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Step 3: Format columns (Convert categorical variables, etc.)


def format_columns(df):
    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return label_encoders

# Analyze data to separate categorical and numerical columns


def analyze_data(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols

# Suggest graph types based on the columns


def graph_suggester(df, categorical_cols, numerical_cols):
    suggestions = {}
    for col in categorical_cols:
        if df[col].nunique() <= 10:
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

# Add a linear regression trendline to a plot


def add_trendline(fig, x_data, y_data, color='red'):
    if len(x_data) == len(y_data):  # Ensure lengths are equal before fitting
        z = np.polyfit(x_data, y_data, 1, '--')
        p = np.poly1d(z)
        trendline_y = p(x_data)
        fig.add_scatter(x=x_data, y=trendline_y, mode='lines',
                        name='Trend Line', line=dict(color=color))

# Visualize the data with random selection


def visualize_data(df, graph_suggestions):
    for col, graph_types in graph_suggestions.items():
        graph_type = random.choice(graph_types)

        if graph_type == 'Bar plot':
            fig = px.bar(df, x=col, y=col if df[col].nunique(
            ) > 10 else df.index, title=f"Bar plot of {col}")

        elif graph_type == 'Count Plot':
            fig = px.histogram(df, x=col, title=f"Count Plot of {col}")

        elif graph_type == 'Pie chart':
            fig = px.pie(df, names=col, title=f"Pie Plot of {col}")

        elif graph_type == 'Histogram':
            fig = px.histogram(df, x=col, title=f"Distribution Plot of {col}")
            # Use value counts for trendline, ensuring x and y lengths are consistent
            value_counts = df[col].value_counts()
            add_trendline(fig, value_counts.index, value_counts.values)

        elif graph_type == 'Boxplot':
            fig = px.box(df, y=col, title=f"Boxplot Plot of {col}")

        elif graph_type == 'Scatterplot':
            num_col = next((n for n in df.select_dtypes(
                include=["number"]).columns if n != col), None)
            if num_col:
                fig = px.scatter(df, x=col, y=num_col,
                                 title=f"Scatter Plot of {col} vs {num_col}")
                # Ensure no NaN values are passed for x and y in scatter plot
                non_nan_data = df[[col, num_col]].dropna()
                add_trendline(fig, non_nan_data[col], non_nan_data[num_col])

        elif graph_type in ['Time series plot', 'Line chart']:
            fig = px.line(df, x=df.index, y=col, title=f"Time series of {col}")

        fig.show()

# Step 4: Get statistics


def get_statistics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    return df[num_cols].describe().T[['mean', '50%', 'std']]


def main():
    parser = argparse.ArgumentParser(
        description="Automatically clean data, suggest stats, and generate graphs.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    # Load and clean data
    df = read_data(args.file_path)
    clean_data(df)

    # Format columns
    label_encoders = format_columns(df)

    # Display stats
    stats = get_statistics(df)
    print("Statistics of Numerical Columns:")
    print(stats)

    # Analyze and visualize
    categorical_cols, numerical_cols = analyze_data(df)
    graph_suggestions = graph_suggester(df, categorical_cols, numerical_cols)
    visualize_data(df, graph_suggestions)


if __name__ == "__main__":
    main()
