import numpy as np
import matplotlib.pyplot as plt

# Function to fill a triangle in a 2D NumPy array


def fill_triangle(canvas, coords):
    x, y = coords.T
    vertices = np.column_stack((x, y))
    fill_coords = np.round(vertices).astype(int)
    canvas.fill_poly(fill_coords, color=1)


# Example usage
canvas_shape = (10, 10)
canvas = np.zeros(canvas_shape)

# Define three coordinates of the triangle
triangle_coords = np.array([[2, 2], [8, 2], [5, 8]])

# Fill the triangle on the canvas
fill_triangle(canvas, triangle_coords)

# Display the result
plt.imshow(canvas, cmap='viridis', origin='lower')
plt.scatter(*triangle_coords.T, color='red',
            marker='o', label='Triangle Vertices')
plt.legend()
plt.show()
