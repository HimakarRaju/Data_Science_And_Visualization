def is_palindrome(arr):
    String = input("Enter a string\n")
    inverse_string = reversed(String)
    if list(String) == list(inverse_string):
        print("The string is a palindrome\n")
    else:
        print("The string is not a palindrome\n")
