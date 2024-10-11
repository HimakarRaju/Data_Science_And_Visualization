from mermaid import Mermaid

Mermaid(
    """%%{
    init: {
        'theme': 'base',
        'themeVariables': {
        'primaryColor': '#BB2528',
        'primaryTextColor': '#fff',
        'primaryBorderColor': '#7C0000',
        'lineColor': '#F8B229',
        'secondaryColor': '#006100',
        'tertiaryColor': '#fff'
        }
    }
    }%%
    graph LR
    A[Gender Guesser] --> B(Load Data)
    B(Load Data) --> C(Train Data)
    C(Train Data)--> D{Ask Predict or Train}
    D{Ask Predict or Train} --> |Predict| E(Predict) --> F(Ask user for name)
    F(Ask user for name) --> G(If can be predicted) --> H(Give Output as male or female or neutral) -->I(ask to the user another name or exit)
    F(Ask user for name) --> J(If cannot be predicted) --> K(check the gender logic)  --> L(ask user confirm name add to data) --> I(ask to the user another name or exit)
    D{Ask Predict or Train} --> |Train| M(Train) --> N(Ask user for name and gender) --> O(save it to pickle file) -->  B(Load Data)
"""
)

mermaid.write("gender_guesser_diagram.png")
