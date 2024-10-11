import os
import subprocess


def get_installed_libraries():
    with open("installed_packages.txt", "w") as f:
        subprocess.run(
            ["pip", "list"], stdout=f, creationflags=subprocess.CREATE_NO_WINDOW
        )
    with open("installed_packages.txt", "r") as f:
        return {line.split()[0] for line in f.readlines()[2:]}


# Define a function to extract library names from a python file
def get_libraries_from_file(file_path):
    libraries = set()
    with open(file_path, "r") as s:
        for line in s.readlines():
            line = line.strip()
            if line.startswith("import"):
                libraries.add(line.split()[1].split(".")[0].lower())
            elif line.startswith("from"):
                words = line.split()
                if "import" in words:
                    libraries.add(words[1].split(".")[0].lower())
    return libraries


# Define a function to get all missing libraries
def get_install_missing(file_path, missing_libraries, failed_installs):
    libraries = get_libraries_from_file(file_path)
    installed_libraries = get_installed_libraries()
    missing = libraries - installed_libraries
    if missing:
        missing_libraries.update(missing)
        failed_installs[file_path] = missing


def install_missing_libraries(missing_libraries, failed_installs):
    with open("errors.txt", "w") as f:
        for file, libs in failed_installs.items():
            for lib in libs:
                install_status = subprocess.run(
                    ["pip", "install", lib], creationflags=subprocess.CREATE_NO_WINDOW
                )
                if install_status.returncode != 0:
                    f.write(f"Error installing {lib} in {file}\n")
                else:
                    print(f"Installed {lib} successfully")


# Main function
def main():
    user_input = input("Enter a Python file/Folder path: ")
    missing_libraries = set()
    failed_installs = {}

    if os.path.isdir(user_input):

        def get_files_recursive(directory):
            files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isdir(file_path):
                    files.extend(get_files_recursive(file_path))
                elif file_path.endswith(".py"):
                    print(f"processing {file_path}")
                    files.append(file_path)
            return files

        files = get_files_recursive(user_input)
        for file in files:
            get_install_missing(file, missing_libraries, failed_installs)

    elif os.path.isfile(user_input) and user_input.endswith(".py"):
        get_install_missing(user_input, missing_libraries, failed_installs)

    else:
        print("Invalid input. Please enter a valid Python file path.")
        return

    if missing_libraries:
        print("Missing libraries: ", missing_libraries)
        install = input("Do you want to install missing libraries?(y/n): ")
        if install.lower() == "y":
            install_missing_libraries(missing_libraries, failed_installs)
            print("Missing libraries installed.")
        else:
            print("Missing libraries not installed.")
    else:
        print("No Missing libraries.")

    print("Summary:")
    print("Missing libraries: ", missing_libraries)
    print("Files with errors: ")
    for f in  failed_installs:
        print(f.split()[-1])

if __name__ == "__main__":
    main()
