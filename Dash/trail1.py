import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Z', 'Y', 'Z'],
    'Value1': [10, 15, 12, 14, 10, 13, 18, 17],
    'Value2': [5.4, 3.3, 5.7, 4.4, 2.5, 6.1, 4.8, 5.0]
}
df = pd.DataFrame(data)

# Initialize Dash app
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Interactive Dashboard with Dash and Plotly"),

    # Dropdown to select categorical column
    html.Label("Select Categorical Column"),
    dcc.Dropdown(
        id='categorical-column',
        options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['object']).columns],
        value='Category',  # Default value
        clearable=False
    ),

    # Dropdown to select numerical column
    html.Label("Select Numerical Column"),
    dcc.Dropdown(
        id='numerical-column',
        options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['number']).columns],
        value='Value1',  # Default value
        clearable=False
    ),

    # Graph Output
    dcc.Graph(id='bar-graph')
])

# Callback to update graph based on user input
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('categorical-column', 'value'),
     Input('numerical-column', 'value')]
)
def update_graph(categorical_col, numerical_col):
    fig = px.bar(df, x=categorical_col, y=numerical_col, color=categorical_col,
                 title=f'Bar plot of {numerical_col} by {categorical_col}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
