import os
import ast


def check_syntax(file_path):
    try:
        with open(file_path, 'r') as file:
            source = file.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, e


def check_files_in_folder(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                result, error = check_syntax(file_path)
                results.append((file_path, result, error))
    return results


def print_results(results):
    for file_path, result, error in results:
        if result:
            print(f'No syntax errors in {file_path}')
        else:
            print(f'Syntax error in {file_path}: {error}')


if __name__ == '__main__':
    path = input("Enter the path of the Python file or folder: ").strip()
    if os.path.isfile(path) and path.endswith('.py'):
        result, error = check_syntax(path)
        if result:
            print(f'No syntax errors in {path}')
        else:
            print(f'Syntax error in {path}: {error}')
    elif os.path.isdir(path):
        results = check_files_in_folder(path)
        print_results(results)
    else:
        print("Please enter a valid Python file or directory path.")
