import numpy as np
from maths_utils import *


def draw_triangle(scrBuf, triVertexs, depthBuffer, draw_func, *args, **kwargs):
    """
    Apply a vector operation function to each vector in the mesh.

    Parameters:
    - mesh: 3D mesh as a NumPy array
    - operation_func: Function to apply to each vector in the mesh
    - *args, **kwargs: Additional arguments to pass to the operation function

    Returns:
    - Resulting mesh after applying the operation
    """

    avg_z = np.sum(triVertexs[..., 2]) / 3
    flat_triangle = triVertexs[..., :2]
    center = centroid_of_triangle(flat_triangle)
    if depthBuffer[center[0], center[1]] < avg_z:
        return scrBuf

    draw_func(scrBuf, flat_triangle, *args, **kwargs)
    depthBuffer[scrBuf > 0] = avg_z

    # draw_func(scrBuf, triVertexs[..., :2], *args, **kwargs)


def draw_line(scrBuf, row1, col1, row2, col2, colorDepth=1):
    """
    draw a 2d line. Color all of the pixels on the line black
    TODO: find a better way to draw lines
    """

    num_points = int(max(abs(col2 - col1), abs(row2 - row1))) + 1
    r_values = np.linspace(int(row1), int(row2), num_points, dtype=int)
    c_values = np.linspace(int(col1), int(col2), num_points, dtype=int)

    scrBuf[r_values, c_values] = colorDepth
    scrBuf[scrBuf < 0] = 0
    scrBuf[scrBuf > 1] = 1


def draw_triangle_filled(scrBuf, triVertexs, colorDepth=1):
    """
    fills a triangle bounded by trivertexs using the fast way

    triVertexs in the form of [row, col]

    code is rewritten from:
    http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    """

    def fillBottomFlatTriangle(v1, v2, v3):

        invslope1 = (v2[1]-v1[1]) / (v2[0]-v1[0])
        invslope2 = (v3[1]-v1[1]) / (v3[0]-v1[0])

        curcol1 = v1[1]
        curcol2 = v1[1]

        minRow = np.floor(v1[0]).astype(int)
        maxRow = np.ceil(v2[0]).astype(int)

        for scanlineY in range(minRow, maxRow+1):
            draw_line(scrBuf, scanlineY, int(curcol1),
                      scanlineY, int(curcol2), colorDepth)
            curcol1 += invslope1
            curcol2 += invslope2

    def fillTopFlatTriangle(v1, v2, v3):
        invslope1 = (v3[1]-v1[1]) / (v3[0]-v1[0])
        invslope2 = (v3[1]-v2[1]) / (v3[0]-v2[0])

        curcol1 = v3[1]
        curcol2 = v3[1]

        minRow = np.floor(v1[0]).astype(int)
        maxRow = np.ceil(v3[0]).astype(int)
        for scanlineY in range(maxRow, minRow-1, -1):
            draw_line(scrBuf, scanlineY, int(curcol1),
                      scanlineY, int(curcol2), colorDepth)
            curcol1 -= invslope1
            curcol2 -= invslope2

    # sort the vertexs by row
    triVertexs = triVertexs[triVertexs[:, 0].argsort()]

    r0 = triVertexs[0][0]
    c0 = triVertexs[0][1]
    r1 = triVertexs[1][0]
    c1 = triVertexs[1][1]
    r2 = triVertexs[2][0]
    c2 = triVertexs[2][1]

    if r1 == r2:
        fillBottomFlatTriangle(triVertexs[0], triVertexs[1], triVertexs[2])
    elif r0 == r1:
        fillTopFlatTriangle(triVertexs[0], triVertexs[1], triVertexs[2])
    else:

        v4row = r1
        v4col = int(c0 + (c2-c0) / (r2-r0) * (r1-r0))

        v4 = np.array([v4row, v4col])
        fillBottomFlatTriangle(triVertexs[0], triVertexs[1], v4)
        fillTopFlatTriangle(triVertexs[1], v4, triVertexs[2])

# def draw_triangle_filled(scrBuf, triVertexs, colorDepth, colorDepth_func=None):
#     """
#     fills a face of triangle with the specified corlor
#     colorDepth func is a function that takes in the coords of the pixel
#         colorDepth_func(np.array[row],np.array[col]) -> np.array[colorDepth]


#     This function fills triangle using barycentric coordinates, which is slow.
#     """

#     # get the coords
#     min_row = int(np.min(triVertexs[..., 0]))
#     max_row = int(np.max(triVertexs[..., 0]))
#     min_col = int(np.min(triVertexs[..., 1]))
#     max_col = int(np.max(triVertexs[..., 1]))

#     # TODO: this is a temporary fix for drawing out of bound
#     min_row = np.maximum(min_row, 0)
#     max_row = np.minimum(max_row, scrBuf.shape[0] - 1)
#     min_col = np.maximum(min_col, 0)
#     max_col = np.minimum(max_col, scrBuf.shape[1] - 1)

#     row_coords = np.arange(min_row, max_row + 1)
#     col_coords = np.arange(min_col, max_col + 1)

#     rows, cols = np.meshgrid(row_coords, col_coords)
#     coords_arr = np.vstack((rows.ravel(), cols.ravel())).T

#     # calculate a mask for the triangle
#     try:
#         w_row, w_col = get_barycentric_coords(triVertexs, coords_arr)
#     except np.linalg.LinAlgError:
#         """
#         TODO:
#             When x1 x2 x3 or y1 y2 y3 line up, the area of triangle equals zero
#             and calulating inverse of the matrix will give a LinAlgError.
#             Add something to detect the situation and simply draw a line.
#         """
#         linebuf = draw_triangle_wireframe(scrBuf, triVertexs, colorDepth)
#         return linebuf

#     mask = coords_arr[(w_row >= 0) & (w_col >= 0) & (w_row + w_col <= 1)]

#     mask_rows = mask[..., 0]
#     mask_cols = mask[..., 1]
#     # TODO: remove this
#     np.append(mask_rows, mask_rows[-1] + 1)
#     np.append(mask_cols, mask_cols[-1] + 1)

#     if colorDepth_func is None:
#         scrBuf[mask_rows, mask_cols] = colorDepth
#     else:
#         scrBuf[mask_rows, mask_cols] = colorDepth_func(mask_rows, mask_cols)

#     return scrBuf


def draw_triangle_wireframe(scrBuf, triVertexs, colorDepth=1):
    """
    takes in the screen buffer (2d ndarry),
    and a matrix[vertex_no][row, col]

    draws the triangle on the screenwith lines interconnecting it
    """
    draw_line(scrBuf, triVertexs[0, 0], triVertexs[0, 1],
              triVertexs[1, 0], triVertexs[1, 1], colorDepth)
    draw_line(scrBuf, triVertexs[0, 0], triVertexs[0, 1],
              triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    draw_line(scrBuf, triVertexs[1, 0], triVertexs[1, 1],
              triVertexs[2, 0], triVertexs[2, 1], colorDepth)
