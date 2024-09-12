def demonstrate_string_operations():
    # Sample string
    sample_str = "We are having fun in this class, Hello World"
    print(f"Original String: '{sample_str}'")

    # 1. Concatenation
    concatenated_str = sample_str + " How are you?"
    print(f"Concatenated String: '{concatenated_str}'")

    # 2. Slicing
    sliced_str = sample_str[0:5]  # Get the first 5 characters
    print(f"Sliced String (first 5 characters): '{sliced_str}'")

    # 3. Length of String
    length = len(sample_str)
    print(f"Length of String: {length}")

    # 4. Convert to Uppercase
    upper_str = sample_str.upper()
    print(f"Uppercase String: '{upper_str}'")

    # 5. Convert to Lowercase
    lower_str = sample_str.lower()
    print(f"Lowercase String: '{lower_str}'")

    # 6. Find a Substring
    find_index = sample_str.find("World")
    print(f"Index of 'World': {find_index}")

    # 7. Replace a Substring
    replaced_str = sample_str.replace("World", "Universe")
    print(f"Replaced String: '{replaced_str}'")

    # 8. Check if String Starts or Ends with a Substring
    starts_with_hello = sample_str.startswith("Hello")
    ends_with_world = sample_str.endswith("World!")
    print(f"Starts with 'Hello': {starts_with_hello}")
    print(f"Ends with 'World!': {ends_with_world}")

    # 9. Split String into List
    split_str = sample_str.split(", ")
    print(f"Split String into List: {split_str}")

    # 10. Join List into String
    joined_str = " - ".join(split_str)
    print(f"Joined String: '{joined_str}'")

    # 11. Strip Whitespace
    whitespace_str = "  Hello, World!   "
    stripped_str = whitespace_str.strip()
    print(f"Stripped String: '{stripped_str}'")
    # 12. Check if String is Numeric
    numeric_str = "12345"
    is_numeric = numeric_str.isnumeric()
    print(f"Is '{numeric_str}' Numeric: {is_numeric}")

    # 13. Check if String is Alphabetic
    alphabetic_str = "Hello"
    is_alphabetic = alphabetic_str.isalpha()
    print(f"Is '{alphabetic_str}' Alphabetic: {is_alphabetic}")

    # 14. String Formatting
    name = "Alice"
    age = 30
    formatted_str = f"My name is {name} and I am {age} years old."
    print(f"Formatted String: '{formatted_str}'")

    # 15. Count Occurrences of a Substring
    count_world = sample_str.count("o")
    print(f"Number of 'o' in '{sample_str}': {count_world}")

    # 16. Reverse a String
    reversed_str = sample_str[::-1]
    print(f"Reversed String: '{reversed_str}'")


demonstrate_string_operations()
