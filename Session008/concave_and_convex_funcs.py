import numpy as np
import matplotlib.pyplot as plt

# Define a convex function: f(x) = x^2


def convex_function(x):
    return x ** 2

# Define a concave function: f(x) = -x^2


def concave_function(x):
    return -x ** 2


# Generate a range of x values
x = np.linspace(-10, 10, 400)

# Calculate y values for both functions
y_convex = convex_function(x)
y_concave = concave_function(x)

# Plot the convex function
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y_convex, label="Convex: $x^2$", color='blue')
plt.title("Convex Function: $f(x) = x^2$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

# Plot the concave function
plt.subplot(1, 2, 2)
plt.plot(x, y_concave, label="Concave: $-x^2$", color='red')
plt.title("Concave Function: $f(x) = -x^2$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
