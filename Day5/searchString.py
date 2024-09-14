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
        lowered_lines = []
        lines = file.readlines()
        for line in lines:
            lowered_lines.append(line.lower())
        return lowered_lines


Active = True


while Active:

    print("Please select to search or replace text ?")
    print("1. Search")
    print("2. Replace")

    User_Input = int(input("Enter 1 or 2 : "))

    if User_Input == 1:

        Data = readFile()
        print(Data)

        user_search_text = input("Enter the search text : ")+"\n"

        for item in Data:
            if user_search_text == item:

                print("found the text {}", user_search_text)

                print("Do you want to replace it ?")
                choice = input("Y or N : ").lower().strip()

                if choice != "y":
                    Active = False

            else:
                print("Text Not Found\n")

    if User_Input == 2:
        Data = readFile()

        print(Data)
        replace_text = input("Enter the text you want to replace : ")
        new_text = input("Enter the new text : ")

        if replace_text in Data:
            Data.replace(replace_text, new_text)
        else:
            print("Invalid operation Try Again")
