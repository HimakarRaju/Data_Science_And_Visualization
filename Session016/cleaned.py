import pytesseract
import cv2
import sympy as sp
import re


# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\System_Installs\Development\Tesseract\tesseract.exe"  # Update with your path
)


def extract_text_from_image(image_path):
    """Extract text from image using pytesseract"""
    try:
        img = cv2.imread(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def create_equation(line):
    """Create a equation from a line of text"""
    try:
        # Check for the presence of '='
        if line.count("=") > 1:
            raise ValueError(
                "Invalid line format. Please ensure the line contains only one '=' sign."
            )

        left_side, right_side = line.split("=") if "=" in line else (line, "")

        # Normalize and prepare sides for sympy
        left_side = left_side.replace(" ", "")  # Remove spaces from left side
        right_side = right_side.replace(" ", "")  # Remove spaces from rightside

        if right_side.strip() == "":  # If no right side, treat it as equal to 0
            right_side = "0"

        return left_side.strip(), right_side.strip()
    except Exception as e:
        print(f"Error creating equation: {e}")
        return "", ""


def fix_multiplication(expr):
    """Insert * between a number and a variable (e.g., 2(x) -> 2*x)"""
    return re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", expr)


def fix_sqrt(expr):
    return expr.replace("sqrt", "sp.sqrt")


def replace_functions(expr):
    """Replace functions like sin, cos, tan with sp.sin, sp.cos, sp.tan"""
    expr = expr.replace("sin", "sp.sin")
    expr = expr.replace("cos", "sp.cos")
    expr = expr.replace("tan", "sp.tan")
    expr = expr.replace("log", "sp.log")
    return expr


def solve_equation(line):
    """Solve an equation using sympy"""
    try:
        x = sp.Symbol("x" or "X")
        X = sp.Symbol("X")
        y = sp.Symbol("y")
        a = fix_sqrt(line)
        b = fix_multiplication(a)
        c = replace_functions(b)

        lhs, rhs = create_equation(c)
        print(f"lhs : {lhs} , rhs : {rhs}")
        if rhs == "0":  # Check if rhs is zero
            rhs = eval(lhs)
            return rhs  # Return lhs as result
        else:
            p = sp.solveset(sp.Eq(eval(lhs), eval(rhs)), x)
            return p
    except Exception as e:
        print(f"Error solving equation: {e}")
        return None


def main(image_path):
    image2 = extract_text_from_image(image_path)
    lines = image2.split("\n")
    for line in lines:
        if line != "":
            print(f"Equation = {line}")
            result = solve_equation(line)
            if result is not None:
                print(f"result: {result}")
                print("\n")


if __name__ == "__main__":
    image_path = "Session016\img4.png"
    main(image_path)
