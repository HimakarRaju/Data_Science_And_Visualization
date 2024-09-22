# matrix = [[2, 3, 4, 5], [4, 5, 6, 7], [9, 8, 7, 6], [3, 2, 6, 7]]
# inputFromUser = int(input(f"select column From 1 - {len(matrix)} : ")) + 1
# print(inputFromUser)

# # Transpose the matrix using zip and convert each row to a list
# cols = [list(col) for col in zip(*matrix)]

# if 0 <= inputFromUser < len(cols):
#     print(f"Column {inputFromUser - 1 }: {cols[inputFromUser]}")
# else:
#     print("Invalid column number. Please choose a valid index.")


# take user input to which column they want to extract
# B = int(input("Enter the number:"))
My_list = [(2, 3, 4, 5), (4, 5, 6, 7), (9, 8, 7, 6), (3, 2, 6, 7),
           (2, 3, 4, 5), (4, 5, 6, 7), (9, 8, 7, 6), (3, 2, 6, 7)]

# c = int(input("Enter the Value : "))
c = 2
count = 0

for row in My_list:
    for i in row:
        if i == c:
            count += 1

print(f"Number of times {c} appears in the list is {count}")