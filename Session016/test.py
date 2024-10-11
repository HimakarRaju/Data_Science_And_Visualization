import pytesseract
from PIL import Image
import sympy as sp


# Configure pytesseract

path_to_tesseract = r"D:\\System_Installs\\Development\\Tesseract\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
# sets the path to the Tesseract executable


def extract_text_from_image(image_path):
    """Extract text from image using pytesseract"""
    try:
        img = Image.open(image_path)  # opens the image file.
        text = pytesseract.image_to_string(img)  # extracts text from the opened image.
        return (
            text.strip()
        )  # returns the extracted text after stripping any leading or trailing whitespace.
    except Exception as e:
        return f"Error extracting text: {e}"


image_path = r"C:\Users\HimakarRaju\Desktop\PreSkilling\Python\Data_Science_And_Visualization\Session016\img3.png"
image2 = extract_text_from_image(image_path)

txt = image2.replace("=", ",")

print(txt)

#  solving the equation
# x = sp.Symbol("x")
# lhs, rhs = txt.split(",")
# print(lhs, rhs)


def sol(txt):
    equations = []
    lines = txt.split("\n")
    print(lines)
    for line in lines:
        line = line.strip()
        print(line)
        if "," in line:
            lhs, rhs = line.split(",")
            print(lhs, rhs)
            x = sp.Symbol("x")
            lhs = lhs.replace("x", "*x")
            print(f"modified lhs: {lhs}")
            try:
                p = sp.solveset(sp.Eq(eval(lhs), eval(rhs)), x)
                print(p)
            except Exception as e:
                print(f"Error solving equation: {e}")
        else:
            print(f"Error: unable to parse equation '{line}'")

    return  # This will return None, you can change it to return something else if needed


sol(txt)
