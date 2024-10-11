import os

installed_libraries = set()
libraries = set()

os.system("pip freeze > installed_packages.txt")

with open('installed_packages.txt', 'r') as f:
    for line in f:
        installed_libraries.add(line.strip().split('==')[0])


def lib_lister(p):
    with open(p, 'r') as s:
        for line in s:
            line = line.strip()
            if line.startswith("import"):
                libs = line.replace("import", "").strip().split(",")
                for lib in libs:
                    libraries.add(lib.strip())
            elif line.startswith("from"):
                lib = line.split(" ")[1]
                libraries.add(lib.split("import")[0].strip())


def main():
    user_input = input("enter a file/ folder path: ")

    if os.path.isfile(user_input):
        if user_input.endswith('.py'):
            lib_lister(user_input)
    elif os.path.isdir(user_input):
        for file in os.listdir(user_input):
            if file.endswith('.py'):
                lib_lister(os.path.join(user_input, file))


# print("Installed libraries:", installed_libraries)
print("Imported libraries:", libraries)


main()
