# import matplotlib.pyplot as plt
# import numpy as np

# x = np.random.rand(50)
# y = np.random.rand(50)
# z = np.polyfit(x, y, 1)
# z1 = np.polyfit(x, y, 2)

# t = np.poly1d(z)
# t1 = np.poly1d(z1)


# plt.scatter(x, y)

# plt.plot(x, t(x), "r--")
# plt.plot(x, t1(x), "g--")


# plt.title("Basic Scatter Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])  # Categories for the bar chart
y = np.array([2, 3, 5, 7, 11])  # Values for the bar chart

# Create bar chart
plt.bar(x, y, color='lightblue', label='Data')

# Calculate trendline (linear regression)
z = np.polyfit(x, y, 1)  # Degree 1 for a linear fit
p = np.poly1d(z)

# Plot trendline
plt.plot(x, p(x), color='red', label='Trendline')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Bar Graph with Trendline')

# Show legend
plt.legend()

# Display plot
plt.show()
