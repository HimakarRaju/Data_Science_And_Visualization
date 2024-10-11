import os

# Define the Mermaid flowchart for the data analysis pipeline
mermaid_code = """
graph TD;
    A[Start] --> B[Load Dataset]
    B --> C[Check Column Names]
    C --> D[Suggest Target Column]
    D --> E[Transform Data]
    E --> F[Evaluate Models]
    F --> G[Generate and Save Plots]
    G --> H[Save Metadata to Pickle]
    H --> I[Log Steps and Errors]
    I --> J[End]
"""

# Save the Mermaid code to a file
output_dir = "C:\\Users\\HimakarRaju\\Desktop\\Milestone2\\DataAnalyser"
mermaid_file_path = os.path.join(output_dir, "data_analysis_pipeline.mmd")
with open(mermaid_file_path, "w") as file:
    file.write(mermaid_code)
