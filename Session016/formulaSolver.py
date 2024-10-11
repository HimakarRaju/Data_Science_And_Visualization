from PIL import Image
import pytesseract
import sympy as sp


# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"D:\System_Installs\Development\Tesseract\tesseract.exe"  # Update with your path
)


def extract_text_from_image(image_path):
    """Extract text from image using pytesseract"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


image_path = r"C:\\Users\\HimakarRaju\\Desktop\\PreSkilling\\Python\\Data_Science_And_Visualization\\Session016\\img2.png"
image2 = extract_text_from_image(image_path)

txt = image2.replace("=", ",")
print(txt)

# # solving the equation
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
                print(f"The required values is : {p}")
            except Exception as e:
                print(f"Error solving equation: {e}")
        else:
            print(f"Error: unable to parse equation '{line}'")
    return  # This will return None, you can change it to return something else if needed


sol(txt)
