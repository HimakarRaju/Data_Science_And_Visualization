import os
import sys
import subprocess
import re

# 1. Store installed Python libraries into installed_libraries
installed_libraries = subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze']).decode('utf-8').splitlines()
installed_libraries = [lib.split('==')[0] for lib in installed_libraries]

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
        imports = re.findall(
            r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import', content)
        imports = [item for sublist in imports for item in sublist if item]
        libraries_in_files.update(imports)

# 4. Compare installed_libraries and libraries_in_files
for lib in libraries_in_files:
    if lib not in installed_libraries:
        missing_libraries.append(lib)

# 5. Install the missing libraries
for lib in missing_libraries:
    print(f"Installing {lib}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])
    print(f"{lib} installed successfully!")
