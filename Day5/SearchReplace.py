# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:02:37 2024

@author: HimakarRaju
"""

# Take user input
# search the string from text file
# Ask the user if he/she wants to replace it
# If YES replace it with the new text given by user
# If NO just stop the program.

"""
#sample text data

apple
guava
mangoes
pineapple
grapes

"""


def readFile():
    with open('TextData.txt', 'r') as file:
        lowered_lines = [line.lower().strip() for line in file.readlines()]  # One-liner for efficiency
    return lowered_lines


def writeFile(data):
    with open('TextData.txt', 'w') as file:
        file.write("\n".join(data))  # More efficient file writing


Active = True

while Active:
    print("Please select an option:")
    print("1. Search")
    print("2. Replace")
    print("3. Delete")
    print("4. Exit")

    User_Input = int(input("Enter 1, 2, 3 or 4: "))

    if User_Input == 1:  # Search and replace option
        Data = readFile()
        user_search_text = input("Enter the search text: ").lower().strip()

        found = False
        for item in Data:
            if user_search_text in item:
                print(f"Found the text: {user_search_text}")
                found = True

                choice = input("Do you want to replace it? (Y or N): ").lower().strip()

                if choice == "y":
                    new_text = input("Enter the new text: ").lower().strip()
                    Data = [line.replace(user_search_text, new_text) for line in Data]
                    writeFile(Data)
                    print(f"Replaced '{user_search_text}' with '{new_text}'.")
                break

        if not found:
            print("Text not found")

    elif User_Input == 2:  # Replace option
        Data = readFile()
        print(Data)
        replace_text = input("Enter the text you want to replace: ").lower().strip()
        new_text = input("Enter the new text: ").lower().strip()

        if any(replace_text in line for line in Data):
            Data = [line.replace(replace_text, new_text) for line in Data]
            writeFile(Data)
            print(f"Replaced '{replace_text}' with '{new_text}'.")
        else:
            print("Text not found. Try again.")

    elif User_Input == 3:  # Delete option
        Data = readFile()
        print(Data)
        delete_text = input("Enter the text you want to delete: ").lower().strip()

        if any(delete_text in line for line in Data):
            Data = [line for line in Data if delete_text not in line]  # Filter out lines containing delete_text
            writeFile(Data)
            print(f"Deleted '{delete_text}'.")
        else:
            print("Text not found. Try again.")

    elif User_Input == 4:  # Exit option
        print("\nExiting the program.")
        Active = False

    else:
        print("Invalid input, please enter 1, 2, 3 or 4.")
