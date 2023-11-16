import numpy as np


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def centroid_of_triangle(vertices):
    # Extract x and y coordinates of vertices
    x_coords, y_coords = vertices[:, 0], vertices[:, 1]

    # Calculate the centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    return np.array([int(centroid_x), int(centroid_y)])


def get_barycentric_coords(triVertexs, coords):
    """
    takes in a 2d triangle in format of [vertex_no][row, col]
    and a list of 2d coords in format of [n][row, col]
    returns the barycentric coords in format of w_row[n], w_col[n]

    Thanks to Prof.McNeily for helping me figure out the math
    """

    v0 = triVertexs[1] - triVertexs[0]
    v1 = triVertexs[2] - triVertexs[0]
    u0 = normalize(v0)
    u1 = normalize(v1)

    # \because   w = (a,b), u1 = (x1,y1), u2 = (x2,y2)
    #            w = w1 + w2 = d1u1 + d2u2
    # \therefore    (a,b) = d1(x1,y1) + d2(x2,y2)
    # \therefore    a = d1x1 + d2x2
    #               b = d1y1 + d2y2
    # \therefore    [a,b] = [[x1,x2],[y1,y2]] [d1,d2]
    # \therefore    [d1,d2] = [[x1,x2],[y1,y2]]^-1 [a,b]
    w = coords - triVertexs[0]
    mat = np.array([[u0[0], u1[0]], [u0[1], u1[1]]])
    d0d1 = np.linalg.inv(mat) @ w.T
    w0 = d0d1[0].reshape(-1, 1) / np.linalg.norm(v0)
    w1 = d0d1[1].reshape(-1, 1) / np.linalg.norm(v1)
    return w0.T.ravel(), w1.T.ravel()
