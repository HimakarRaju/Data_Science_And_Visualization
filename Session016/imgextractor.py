import pytesseract
import sympy
import cv2
import re

imgpath = input("\nEnter image path : ")

path_to_tesseract = r"D:\System_Installs\Development\Tesseract\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract


def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        text = pytesseract.image_to_string(image_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


def process_equations(text):
    equations = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if "=" in line:
            try:
                # Replace 'x' with '*x' to make it a valid Python syntax
                line = re.sub(r"(\d)(x)", r"\1*\2", line)
                lhs, rhs = line.split("=")
                lhs = sympy.sympify(lhs)
                rhs = sympy.sympify(rhs)
                equations.append((lhs, rhs))
            except Exception as e:
                print(f"Error processing equation: {e}")
    return equations


text2 = extract_text_from_image(imgpath)
equations = process_equations(text2)

for equation in equations:
    print(f"Equation: {equation[0]} = {equation[1]}")
    solution = sympy.solve(equation[0] - equation[1], equation[0])
    print(f"Solution: {solution}")
    print()
