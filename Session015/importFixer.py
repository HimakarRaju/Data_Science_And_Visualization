"""
1. store installed python libraries into installed_libraries
2. take user input for file path and if file path is of dir read all python files and then store all the libraries inside the python file and store it to libraries_in_files
3. compare installed_libraries and libraries_in_files
4. if i from libraries_in_files not in installed_libraries append it to missing_libraries
5. install the missing libraries
"""

import os
import pip

# 1. Store installed Python libraries into installed_libraries
installed_libraries = [pkg.key for pkg in pip.get_installed_distributions()]

# 2. Take user input for file path and read all Python files
file_path = input("Enter the file path or directory: ")

if os.path.isdir(file_path):
    python_files = [os.path.join(file_path, f)
                    for f in os.listdir(file_path) if f.endswith('.py')]
else:
    python_files = [file_path]

# Initialize libraries_in_files and missing_libraries
libraries_in_files = set()
missing_libraries = []

# 3. Read all Python files and store all the libraries inside the Python file
for file in python_files:
    with open(file, 'r') as f:
        content = f.read()
        imports = [line.split('import')[1].strip(
        ) for line in content.splitlines() if line.startswith('import')]
        libraries_in_files.update(imports)

# 4. Compare installed_libraries and libraries_in_files
for lib in libraries_in_files:
    if lib not in installed_libraries:
        missing_libraries.append(lib)

# 5. Install the missing libraries
for lib in missing_libraries:
    print(f"Installing {lib}...")
    os.system(f"pip install {lib}")
    print(f"{lib} installed successfully!")
