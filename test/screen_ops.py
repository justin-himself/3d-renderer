import numpy as np
from mathutils import *


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

    result = draw_func(scrBuf, flat_triangle, *args, **kwargs)
    depthBuffer[result > 0] = avg_z
    return result

    # return draw_func(scrBuf, triVertexs[..., :2], *args, **kwargs)


def draw_line(scrBuf, row1, col1, row2, col2, colorDepth=1):
    """
    draw a 2d line. Color all of the pixels on the line black
    TODO: find a better way to draw lines
    """

    num_points = int(max(abs(col2 - col1), abs(row2 - row1))) + 1
    r_values = np.linspace(int(row1), int(row2), num_points, dtype=int)
    c_values = np.linspace(int(col1), int(col2), num_points, dtype=int)

    newScrBuf = np.copy(scrBuf)
    newScrBuf[r_values, c_values] = colorDepth
    newScrBuf[newScrBuf < 0] = 0
    newScrBuf[newScrBuf > 1] = 1

    return newScrBuf


def draw_triangle_filled(scrBuf, triVertexs, colorDepth, colorDepth_func=None):
    """
    fills a face of triangle with the specified corlor
    colorDepth func is a function that takes in the coords of the pixel
        colorDepth_func(np.array[row],np.array[col]) -> np.array[colorDepth]

    https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    """

    # get the coords
    min_row = int(np.min(triVertexs[..., 0]))
    max_row = int(np.max(triVertexs[..., 0]))
    min_col = int(np.min(triVertexs[..., 1]))
    max_col = int(np.max(triVertexs[..., 1]))

    # TODO: this is a temporary fix for drawing out of bound
    min_row = np.maximum(min_row, 0)
    max_row = np.minimum(max_row, scrBuf.shape[0] - 1)
    min_col = np.maximum(min_col, 0)
    max_col = np.minimum(max_col, scrBuf.shape[1] - 1)

    row_coords = np.arange(min_row, max_row + 1)
    col_coords = np.arange(min_col, max_col + 1)

    rows, cols = np.meshgrid(row_coords, col_coords)
    coords_arr = np.vstack((rows.ravel(), cols.ravel())).T

    # calculate a mask for the triangle
    try:
        w_row, w_col = get_barycentric_coords(triVertexs, coords_arr)
    except np.linalg.LinAlgError:
        """
        TODO: 
            When x1 x2 x3 or y1 y2 y3 line up, the area of triangle equals zero
            and calulating inverse of the matrix will give a LinAlgError.
            Add something to detect the situation and simply draw a line.
        """
        linebuf = draw_line(
            scrBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[1, 0], triVertexs[1, 1], colorDepth)
        return linebuf

    mask = coords_arr[(w_row >= 0) & (w_col >= 0) & (w_row + w_col <= 1)]

    mask_rows = mask[..., 0]
    mask_cols = mask[..., 1]

    newSrcBuf = np.copy(scrBuf)
    if colorDepth_func is None:
        newSrcBuf[mask_rows, mask_cols] = colorDepth
    else:
        newSrcBuf[mask_rows, mask_cols] = colorDepth_func(mask_rows, mask_cols)

    return newSrcBuf


def draw_triangle_wireframe(scrBuf, triVertexs, colorDepth=1):
    """
    takes in the screen buffer (2d ndarry),
    and a matrix[vertex_no][row, col]

    draws the triangle on the screenwith lines interconnecting it
    """

    newSrcBuf = scrBuf
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[1, 0], triVertexs[1, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[1, 0], triVertexs[1, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    return newSrcBuf
