import numpy as np


# Example usage with the provided coordinates
triangle_vertices = np.array([[146., 362.],
                              [133., 364.],
                              [148., 364.]])

centroid = centroid_of_triangle(triangle_vertices)
print("Centroid of the triangle:", centroid)
