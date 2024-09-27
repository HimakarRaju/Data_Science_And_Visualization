'''
Start
  |
  v
Input: Data Points (x, y)
  |
  v
Step 1: Calculate Mean of x (mean_x) and y (mean_y)
  |
  v
Step 2: Calculate the numerator (sum_xy) and denominator (sum_xx) for the slope (m):
    sum_xy = Σ((x_i - mean_x) * (y_i - mean_y))
    sum_xx = Σ((x_i - mean_x)²)
  |
  v
Step 3: Compute the slope (m):
    m = sum_xy / sum_xx
  |
  v
Step 4: Compute the y-intercept (b):
    b = mean_y - m * mean_x
  |
  v
Step 5: Define the linear regression line equation:
    y = m * x + b
  |
  v
Step 6: Use the line equation to predict y for new x values
  |
  v
Output: Slope (m), Intercept (b), and Line Equation
  |
  v
End



'''


def linear_regression(x, y):
    # Ensure x and y have the same length
    if len(x) != len(y):
        raise ValueError("The length of x and y must be the same.")
    
    # Number of data points
    N = len(x)
    
    # Calculate the sums
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(N))
    sum_x_squared = sum(x[i] ** 2 for i in range(N))
    
    # Calculate slope (m)
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x ** 2)
    
    # Calculate intercept (b)
    b = (sum_y - m * sum_x) / N
    
    return m, b

# Example usage
x = [1, 2, 3, 4, 5,16,20]
#y = [2, 3, 5, 7, 11,20,21]
y = [-2, -3, -5, -7, -11,-20,-21]
m, b = linear_regression(x, y)
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
