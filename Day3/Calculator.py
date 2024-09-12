# Program for simple Calculator

# Functions for add, sub, mul, div
def addition(num1, num2):
    return num1 + num2


def subtraction(num1, num2):
    return num1 - num2


def multiplication(num1, num2):
    return num1 * num2


def division(num1, num2):
    if num2 == 0:
        return "Cannot divide by zero"
    return num1 / num2


def function_input():
    function = int(input(
        "Enter the number of the operation:\n1. Addition\n2. Subtraction\n3. Multiplication\n4. Division\n"))
    return function


def result(function):
    if function in functions:
        res = functions[function](num1, num2)
    return res


# Main program
num1 = int(input('Enter number 1: '))
num2 = int(input('Enter number 2: '))
function = function_input()

functions = {
    1: addition,
    2: subtraction,
    3: multiplication,
    4: division,
}

if function in functions:
    result_value = result(function)
    print(f'Result: {result_value}')
else:
    print("Invalid operation number\n")
    result(function_input())
