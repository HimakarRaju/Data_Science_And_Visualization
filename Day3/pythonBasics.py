def Add(a, b):
    print(a + b)


def Subtract(a, b):
    print(a - b)


def Multiply(a, b):
    print(a * b)


def Divide(a, b):
    print(a / b)


while True:
    print("Select which operation you want to do:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    # Taking choice from user
    choice = int(input("Enter choice (1/2/3/4):"))
    a = int(input("Enter the first number:"))
    b = int(input("Enter the second number:"))

    # Perform calculations according to user choice
    if choice == 1:
        Add(a, b)
    elif choice == 2:
        Subtract(a, b)
    elif choice == 3:
        Multiply(a, b)
    elif choice == 4:
        Divide(a, b)
    else:
        print("Enter a valid choice")

    TryAgain = input("Do you want to repeat one more? Y/N: ")
    if TryAgain.lower() != "y":
        break
