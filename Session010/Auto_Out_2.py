# Import necessary libraries
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


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


# Analyze data to separate categorical and numerical columns
def analyze_data(df):
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


# Suggest graph types based on the columns
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


def suggested_plots(df, graphs):
    for col, graph_types in graphs.items():
        for graph_type in graph_types:
            if graph_type == 'Bar plot':
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.bar(df, x=df.index, y=col)
                else:
                    fig = px.bar(df, x=col)
                fig.show()
            elif graph_type == 'Count Plot':
                fig = px.countplot(df, x=col)
                fig.show()
            elif graph_type == 'Pie chart':
                fig = px.pie(df, names=col)
                fig.show()
            elif graph_type == 'Heatmap':
                fig = px.density_heatmap(df, x=df.index, y=col)
                fig.show()
            elif graph_type == 'Histogram':
                fig = px.histogram(df, x=col)
                fig.show()
            elif graph_type == 'Boxplot':
                fig = px.box(df, y=col)
                fig.show()
            elif graph_type == 'Scatterplot':
                fig = px.scatter(df, x=df.index, y=col)
                fig.show()
            elif graph_type == 'Time series plot':
                fig = px.line(df, x=df.index, y=col)
                fig.show()
            elif graph_type == 'Line chart':
                fig = px.line(df, x=df.index, y=col)
                fig.show()
    return


# Step 4: Find best column statistics (mean, median, std for numerical columns)
def get_statistics(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    stats = df[num_cols].describe().T
    return stats[['mean', '50%', 'std']]  # '50%' is median


# Visualize the data using advanced graphs from code1
def visualize_data(df, categorical_cols, numerical_cols):
    # Plotting categorical columns
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            fig = px.bar(df, x=col, color=col, title=f"Count Plot of {
                         col}", trendline="ols")
            fig.layout.template = 'presentation'
            fig.show()

    # Plotting numerical columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribution Plot of {
                           col}")
        fig.layout.template = 'presentation'
        fig.show()

    # Scatter plots between categorical and numerical columns
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() <= 10:
                fig = px.scatter(df, x=cat_col, y=num_col, color=cat_col, size=num_col,
                                 hover_data=[cat_col, num_col],
                                 title=f"Scatter Plot of {cat_col} vs {num_col}", trendline="ols", trendline_options=dict(log_x=True))
                fig.layout.template = 'presentation'
                fig.show()

    # Plot a heatmap of correlations for numerical columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Automatically clean data, suggest stats, and generate graphs.")
    parser.add_argument("file_path", type=str,
                        help="Path to the input CSV or Excel file.")
    args = parser.parse_args()

    # Step 1: Load and clean data
    df = read_data(args.file_path)
    df = clean_data(df)

    # Step 2: Format columns
    df, label_encoders = format_columns(df)

    # Step 3: Display stats
    stats = get_statistics(df)
    print("Statistics of Numerical Columns:")
    print(stats)

    # Step 4: Suggest and visualize graphs
    categorical_cols, numerical_cols = analyze_data(df)
    graph_suggestions = graph_suggester(df, categorical_cols, numerical_cols)
    visualize_data(df, categorical_cols, numerical_cols)

    # Step 5: Display graph suggestions
    print("\nSuggested Graphs: ")
    for col, graph in graph_suggestions.items():
        print(f'{col}: {graph}')

    plotter = suggested_plots(df, graph_suggestions)
    for col, plot in plotter.items():
        print(f'{col}: {plot}')
    print(plotter)


if __name__ == "__main__":
    main()
