import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def analyze_data(file_path):
    """
    Performs comprehensive data analysis on a given CSV file.

    Args:
        file_path: Path to the user-provided CSV file.
    """

    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Exploratory Data Analysis (EDA)
        print("Data Summary:\n", data.describe())
        print("\nData Types:\n", data.dtypes)
        print("\nMissing Values:\n", data.isnull().sum())

        # Visualization
        sns.pairplot(data)
        plt.show()

        # Statistical Tests
        for column in data.columns:
            if data[column].dtype == 'float64' or data[column].dtype == 'int64':
                # Normality test
                shapiro_test_stat, shapiro_p_value = shapiro(data[column])
                print(f"Shapiro-Wilk Test for {column}: {shapiro_test_stat}, {shapiro_p_value}")

                # Correlation analysis
                correlation_matrix = data.corr()
                print(correlation_matrix)

                # Hypothesis testing (t-test, ANOVA)
                if len(data.select_dtypes(include=['object']).columns) > 0:
                    # Handle potential presence of a 'group' column
                    if 'group' in data.columns:
                        if len(data['group'].unique()) == 2:
                            t_stat, p_value = ttest_ind(data[data['group'] == 'group1'][column],
                                                       data[data['group'] == 'group2'][column])
                            print(f"T-test for {column}: {t_stat}, {p_value}")
                        else:
                            f_stat, p_value = f_oneway(*[data[column][data['group'] == group] for group in data['group'].unique()])
                            print(f"ANOVA for {column}: {f_stat}, {p_value}")
                    else:
                        print(f"No 'group' column found for hypothesis testing on {column}.")

            elif data[column].dtype == 'object':
                # Chi-square test (for categorical variables)
                categorical_cols = data.select_dtypes(include='object').columns
                if len(categorical_cols) > 1:
                    if column in categorical_cols:
                        contingency_table = pd.crosstab(data[categorical_cols[0]], data[column])
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        print(f"Chi-square Test for {column}: {chi2_stat}, {p_value}")
                    else:
                        print(f"Skipping Chi-square test for {column} as it's not paired with another categorical column.")
                else:
                    print(f"Skipping Chi-square test as there's only one categorical column.")

        # Feature Engineering (handle categorical variables)
        categorical_cols = data.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            data_encoded = pd.get_dummies(data, columns=categorical_cols)
        else:
            data_encoded = data.copy()

        # Modeling (example: linear regression, decision tree, random forest, neural network)
        X = data_encoded.drop('target_column', axis=1)
        y = data_encoded['target_column']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Linear Regression
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        predictions_lr = model_lr.predict(X_test)

        # Decision Tree
        model_dt = DecisionTreeClassifier()
        model_dt.fit(X_train, y_train)
        predictions_dt = model_dt.predict(X_test)

        # Random Forest
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        predictions_rf = model_rf.predict(X_test)

        # Neural Network
        model_nn = MLPClassifier(hidden_layer_sizes=(100, 50))
        model_nn.fit(X_train, y_train)
        predictions_nn = model_nn.predict(X_test)

        # Evaluation
        print("Model Evaluation:")
        print("Linear Regression:")
        print("R-squared:", model_lr.score(X_test, y_test))

        # ... (evaluate other models)

        # Visualization
        plt.scatter(y_test, predictions_lr)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values (Linear Regression)")
        plt.show()

        # ... (visualize other models)

    except Exception as e:
        print(f"Error occurred: {e}")

# Provide the path to your CSV file
file_path = "customer_data.csv"
analyze_data(file_path)
