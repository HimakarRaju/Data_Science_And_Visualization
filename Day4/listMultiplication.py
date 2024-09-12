def multiply_list(my_list):
    val = 1
    for i in my_list:
        val *= i
    return val

my_list = [[1,2,3],[4,5],[10],[22,23,99]]

print("The List values are : "+ str(my_list))

a = multiply_list([b for d in my_list for b in d])

print(a)

