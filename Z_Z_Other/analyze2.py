import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load data from CSV or Excel file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
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

    # Create a line plot or count plot based on the selected column
    if df[non_numeric_col].nunique() <= 10:  # Limit for categorical data
        ax = sns.countplot(data=df, x=non_numeric_col)
        plt.title(f'Count of {non_numeric_col}')
        plt.xlabel(non_numeric_col)
        plt.ylabel('Count')

        # Add data labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
    else:
        ax = sns.lineplot(data=df, x=non_numeric_col,
                          y=numeric_col, marker='o')
        plt.title(f'{numeric_col} vs {non_numeric_col}')
        plt.xlabel(non_numeric_col)
        plt.ylabel(numeric_col)
        plt.tight_layout()
        # Add data labels
        for x, y in zip(df[non_numeric_col], df[numeric_col]):
            ax.annotate(f'{y}', (x, y), textcoords='offset points',
                        xytext=(0, 5), ha='center')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    file_path = input("Enter the file path (CSV or Excel): ")
    df = load_data(file_path)

    # Split columns
    numeric_cols, non_numeric_cols = split_columns(df)

    print("Numeric columns:", numeric_cols)
    print("Non-numeric columns:", non_numeric_cols)

    # Select numeric column
    print("\nSelect a numeric column:")
    for i, col in enumerate(numeric_cols):
        print(f"{i}: {col}")
    numeric_choice = int(
        input("Enter the number corresponding to the numeric column: "))
    numeric_col = numeric_cols[numeric_choice]

    # Select non-numeric column
    print("\nSelect a non-numeric column:")
    for i, col in enumerate(non_numeric_cols):
        print(f"{i}: {col}")
    non_numeric_choice = int(
        input("Enter the number corresponding to the non-numeric column: "))
    non_numeric_col = non_numeric_cols[non_numeric_choice]

    # Plot the graph
    plot_graph(df, numeric_col, non_numeric_col)


if __name__ == "__main__":
    main()
