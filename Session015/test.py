import os

installed_libraries = []
libraries = []

os.system("pip freeze > installed_packages.txt")

with open('installed_packages.txt', 'r') as f:
    installed_libraries = [t.strip().split('==')[0] for t in f.readlines()]

print("Installed libraries:", installed_libraries)


def lib_lister(file_path):
    with open(file_path, 'r') as s:
        ts = s.readlines()
        for t in ts:
            t = t.strip()
            if t.startswith('import '):
                libraries.append(t.split()[1])
            elif t.startswith('from '):
                text = t.split(" ")
                if "import" in text:
                    libraries.append(t.index("import")+1)

                # libraries.append(t.split()[1])


user_input = input('Enter a Python file path: ')

if os.path.isfile(user_input):
    if user_input.endswith('.py'):
        lib_lister(user_input)
    else:
        print("The provided file is not a Python file.")
else:
    print("Invalid input. Please enter a valid Python file path.")

missing_libraries = [
    lib for lib in libraries if lib not in installed_libraries]

if missing_libraries:
    print("Missing libraries:", missing_libraries)
else:
    print("No missing libraries.")
