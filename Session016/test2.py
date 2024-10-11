def create_equation(formula):
    # Check for the presence of '='
    if formula.count("=") > 1:
        raise ValueError(
            "Invalid formula format. Please ensure the formula contains only one '=' sign."
        )

    left_side, right_side = formula.split("=") if "=" in formula else (formula, "")

    # Normalize and prepare sides for sympy
    left_side = left_side.replace(" ", "")  # Remove spaces from left side
    right_side = right_side.replace(" ", "")  # Remove spaces from right side

    if not right_side:  # If no right side, treat it as equal to zero
        right_side = "0"

    return left_side.strip(), right_side.strip()
