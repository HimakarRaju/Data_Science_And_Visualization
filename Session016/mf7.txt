import cv2
import pytesseract
import sympy as sp

# Fix missing multiplication like 2(x) -> 2*x
def fix_multiplication(expr):
    import re
    # Insert * between a number and a variable (e.g., 2(x) -> 2*x)
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
    return expr

# Convert degrees to radians for trigonometric functions using sympy
def evaluate_trig(expr):
    try:
        return sp.sympify(expr, evaluate=False).evalf(subs={sp.pi: sp.pi.evalf()})
    except Exception as e:
        print(f"Error in evaluating trigonometric expression: {e}")
        return None

def extract_formula(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return None

    # Use Tesseract to do OCR on the image
    raw_output = pytesseract.image_to_string(image)
    print(f"Raw OCR output: '{raw_output}'")

    # Extract formulas by splitting by new lines and filtering empty lines
    formulas = [line for line in raw_output.splitlines() if line.strip()]
    return formulas

def show_trigonometric_steps(left_side, right_side):
    solutions = None  # Initialize solutions to None
    try:
        # Sympify the sides of the equation
        left_expr = sp.sympify(left_side)
        right_expr = sp.sympify(right_side)

        # Display the initial equation
        print(f"Step 1: Initial Equation: {left_expr} = {right_expr}")

        # Check if it's a direct evaluation case (e.g., sin(45))
        if left_expr.has(sp.Symbol('x')) or right_expr.has(sp.Symbol('x')):
            # Get the free symbols (variables) involved in the equation
            free_symbols = left_expr.free_symbols.union(right_expr.free_symbols)

            if len(free_symbols) == 1:
                # Solve for the only variable present
                var = list(free_symbols)[0]
            else:
                # If there are multiple variables, default to solving for x
                var = sp.Symbol('x') if sp.Symbol('x') in free_symbols else list(free_symbols)[0]
                print(f"Solving for {var}")

            # Apply SymPy's solving mechanism for trigonometric equations
            solutions = sp.solve(sp.Eq(left_expr, right_expr), var)

            # Display the general solution
            print(f"Step 2: Solve for {var}: Found solutions.")

            if not solutions:
                # If no solutions, evaluate and print the left expression value
                evaluated_result = left_expr.evalf()
                print(f"No solutions exist. Evaluated result: {evaluated_result}")
            else:
                # Display only the top 3 solutions
                top_solutions = solutions[:3]
                for i, solution in enumerate(top_solutions):
                    print(f"Solution {i + 1}: {solution}")

                # Step-by-step evaluation
                print(f"\nStep 3: Evaluating each solution:")
                for i, sol in enumerate(top_solutions):
                    eval_left = left_expr.subs(var, sol).evalf()
                    eval_right = right_expr.subs(var, sol).evalf()
                    print(f"  Solution {i + 1}: {sol} -> Left: {eval_left}, Right: {eval_right}")

        else:
            # If there are no variables, just evaluate the left expression
            evaluated_result = left_expr.evalf()
            print(f"Evaluated result: {evaluated_result}")

        return solutions
    
    except Exception as e:
        print(f"Error showing steps for equation: {e}")
        return None

def show_logarithmic_steps(left_side, right_side):
    solutions = None  # Initialize solutions to None
    try:
        # Sympify the sides of the equation
        left_expr = sp.sympify(left_side)
        right_expr = sp.sympify(right_side)

        # Display the initial equation
        print(f"Step 1: Initial Equation: {left_expr} = {right_expr}")

        # Check if it's a direct evaluation case (e.g., log(10))
        if left_expr.has(sp.Symbol('x')) or right_expr.has(sp.Symbol('x')):
            # Get the free symbols (variables) involved in the equation
            free_symbols = left_expr.free_symbols.union(right_expr.free_symbols)

            if len(free_symbols) == 1:
                # Solve for the only variable present
                var = list(free_symbols)[0]
            else:
                # If there are multiple variables, default to solving for x
                var = sp.Symbol('x') if sp.Symbol('x') in free_symbols else list(free_symbols)[0]
                print(f"Solving for {var}")

            # Apply SymPy's solving mechanism for logarithmic equations
            solutions = sp.solve(sp.Eq(left_expr, right_expr), var)

            # Display the general solution
            print(f"Step 2: Solve for {var}: Found solutions.")

            if not solutions:
                # If no solutions, evaluate and print the left expression value
                evaluated_result = left_expr.evalf()
                print(f"No solutions exist. Evaluated result: {evaluated_result}")
            else:
                # Display only the top 3 solutions
                top_solutions = solutions[:3]
                for i, solution in enumerate(top_solutions):
                    print(f"Solution {i + 1}: {solution}")

                # Step-by-step evaluation
                print(f"\nStep 3: Evaluating each solution:")
                for i, sol in enumerate(top_solutions):
                    eval_left = left_expr.subs(var, sol).evalf()
                    eval_right = right_expr.subs(var, sol).evalf()
                    print(f"  Solution {i + 1}: {sol} -> Left: {eval_left}, Right: {eval_right}")

        else:
            # If there are no variables, just evaluate the left expression
            evaluated_result = left_expr.evalf()
            print(f"Evaluated result: {evaluated_result}")

        return solutions
    
    except Exception as e:
        print(f"Error showing steps for logarithmic equation: {e}")
        return None

def create_equation(formula):
    if formula.count('=') > 1:
        raise ValueError("Invalid formula format. Please ensure the formula contains only one '=' sign.")

    left_side, right_side = formula.split('=') if '=' in formula else (formula, '')

    left_side = left_side.replace(' ', '')
    right_side = right_side.replace(' ', '')

    if not right_side:
        right_side = '0'

    return left_side.strip(), right_side.strip()

def evaluate_or_solve(left_side, right_side):
    try:
        left_side = fix_multiplication(left_side)
        right_side = fix_multiplication(right_side)

        left_expr = sp.sympify(left_side)
        right_expr = sp.sympify(right_side)

        # Check for trigonometric functions and solve
        if any(trig in left_side for trig in ['sin', 'cos', 'tan']):
            print("Recognized a trigonometric equation.")
            return show_trigonometric_steps(left_side, right_side)

        # Check for logarithmic functions and solve
        if any(log in left_side for log in ['log', 'ln']):
            print("Recognized a logarithmic equation.")
            return show_logarithmic_steps(left_side, right_side)

        # If no variables are present, evaluate as a numeric expression
        if not left_expr.free_symbols:
            result = left_expr - right_expr
            return f"Evaluated result: {result.evalf()}"

        equation = sp.Eq(left_expr, right_expr)
        solution = sp.solve(equation)
        return solution

    except Exception as e:
        print(f"Error creating equation: {e}")
        return None

def main():
    # Input image file path
    image_path = input("Please enter the image file path: ")
    print("Original image loaded successfully.")

    # Extract formulas from the image
    formulas = extract_formula(image_path)
    if not formulas:
        print("No formulas extracted.")
        return

    for formula in formulas:
        print(f"Extracted formula: {formula}")
        
        try:
            # Create equation from formula
            left_side, right_side = create_equation(formula)
            print(f"Left side before sympify: '{left_side}'")
            print(f"Right side before sympify: '{right_side}'")

            # Solve or evaluate the equation
            solution = evaluate_or_solve(left_side, right_side)
            
            if solution is not None:
                print(f"Solution: {solution}")
            else:
                print("Solution: None")
        
        except ValueError as ve:
            print(ve)
            print("Solution: None")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Solution: None")

if __name__ == "__main__":
    main()
