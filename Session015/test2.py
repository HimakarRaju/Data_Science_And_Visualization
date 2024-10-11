import os


def get_installed_libraries():
    os.system("pip list > installed_packages.txt")
    with open('installed_packages.txt', 'r') as f:
        return {line.split()[0].lower() for line in f.readlines()[2:]}


def get_libraries_from_file(file_path):
    libraries = set()
    with open(file_path, 'r') as s:
        for line in s.readlines():
            line = line.strip()
            if line.startswith('import'):
                libraries.add(line.split()[1].split('.')[0].lower())
            elif line.startswith('from'):
                words = line.split()
                if 'import' in words:
                    libraries.add(words[1].split('.')[0].lower())
    return libraries


user_input = input('Enter a Python file path: ')


if os.path.isdir(user_input):
    files = [os.path.join(user_input, file)
             for file in os.listdir(user_input) if file.endswith(".py")]
    for file in files:
        libraries = get_libraries_from_file(file)
        print(f"Required libraries for {file}: {libraries}")

        installed_libraries = get_installed_libraries()
        print("Installed libraries:", installed_libraries)

        missing_libraries = libraries - installed_libraries

        if missing_libraries:
            print("Missing libraries:", missing_libraries)
            install = input(
                "Do you want to install missing libraries? (y/n): ")
            if install.lower() == 'y':
                for lib in missing_libraries:
                    os.system(f"pip install {lib}")
                print("Missing libraries installed.")

                installed_libraries = get_installed_libraries()
                print("Updated installed libraries:", installed_libraries)

        else:
            print("No missing libraries.")
else:
    if os.path.isfile(user_input) and user_input.endswith('.py'):
        libraries = get_libraries_from_file(user_input)
        print("Required libraries:", libraries)

        installed_libraries = get_installed_libraries()
        print("Installed libraries:", installed_libraries)

        missing_libraries = libraries - installed_libraries

        if missing_libraries:
            print("Missing libraries:", missing_libraries)
            install = input(
                "Do you want to install missing libraries? (y/n): ")
            if install.lower() == 'y':
                for lib in missing_libraries:
                    os.system(f"pip install {lib}")
                print("Missing libraries installed.")

                installed_libraries = get_installed_libraries()
                print("Updated installed libraries:", installed_libraries)

        else:
            print("No missing libraries.")
    else:
        print("Invalid input. Please enter a valid Python file path.")
