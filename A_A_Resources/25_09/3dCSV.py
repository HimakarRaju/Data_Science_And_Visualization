import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('customer_data.csv')

# Assuming your data has columns 'x', 'y', 'z', and 'variation'
x = df['age']
y = df['spending']
z = df['purchase_frequency']
variations = df['purchase_frequency']

# Create a colormap from red to orange
colors = ['red', 'orange']
cmap = ListedColormap(colors)

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot scatter points with unique colors based on variations
scatter = ax.scatter(x, y, z, c=variations, cmap=cmap)

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a colorbar
plt.colorbar(scatter)

# Show plot
plt.show()
