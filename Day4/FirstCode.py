# print("Hello World")

# => Assigning same string to multiple variables
a = b = c = d = e = "My first code in python"

# L.H.S == R.H.S
var = [1, 2, 3]  # List initialization / creating list
a, b, c = var  # UNPACKING

friends = ["shiva", "vinay"]
a, b = friends  # UNPACKING
print(a, b)

x = 0


def code():
    x = 10
    print(f"local x : {x}")


print(f"global x: {x}")
code()


num1 = 120
num2 = 4000


def sum_numbers():
    sum_num = num1+num2
    print(sum_num)
    return sum_num, 1


print(sum_numbers())  # when return value is used
# it prints print inside function as well as it will print return value

sum_numbers()  # if function is called and has print in it we don't need to use return

# return is used when we want to use the value returned by the function in our code
