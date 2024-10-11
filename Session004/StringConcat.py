# Given list of names
Names = [["chandrashekar", "pavan"], ["surendra", "premchand"], ["are good"]]

# output should be
# chandrashekar surendra are good
# pavan premchand are good

# Iterate through each sublist
# for listItem in Names:
#     # Concatenate the first and second names with the third item
#     if len(listItem) > 1:
#         Mixed_Name = listItem[0]+" "+listItem[1]+" "+Names[2][0]
#         print(Mixed_Name)

Flattened_List = []
for listItem in Names:
    if len(listItem) > 1:
        for i in listItem:
            Flattened_List.append(i)

mixed_
print(Flattened_List)
for i in range(len(Flattened_List)):
    if i % 2 != 0:

        print(Flattened_List[i], end=" ")
    else:
        print("\n")
        print(Flattened_List[i], end=" ")
