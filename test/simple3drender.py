import concurrent.futures as cf
import numpy as np
from numpy import cos as c, sin as s
import cv2
import time
import queue
import os

# animate_by_maplotlib(lambda x: np.random.random((200, 300)), range(1, 100), 30)
screenHeight = 480
screenWidth = 720
aspectRatio = screenHeight / screenWidth
fov = 90.0 / 360 * 2 * np.pi
far = 1000
near = 1

# THREAD_NUMBER = min(32, os.cpu_count() + 4)
# # max number of frames to be buffered
# FRAME_BUFFER_SIZE = 1024

frame = np.zeros((screenHeight, screenWidth))
# frame_buffer = np.zeros((0, screenHeight, screenWidth))


def load_obj(file_path):
    """
    take object file and output a matrix descriping 3d mesh,
    in the format of 

        matrix[tri_no][vertex_no][x, y, z, w]

    Where always equals to 1. (Homogeneous Coordinates)
    Plogon faces will be converted into component triangles.
    """

    with open(file_path) as obj_file:
        obj_lines = obj_file.readlines()

    # rules:
        # v x y z       - vertex with –oord x,y,z
        # f a b c...    - face with vertex indexed a b c

    # find starting idx of vertexs
    vertex_line_base_idx = 0
    for idx, line in enumerate(obj_lines):
        if line[:2] == "v ":
            vertex_line_start_idx = idx
            break

    # load object into a matrix like
    # matrix[tri_no][vertex_no][x, y, z, w]
    obj_tri_faces = np.empty((0, 3, 4))
    for line in obj_lines:
        if line[:2] != "f ":
            continue
        vertexs = [int(v.split("/")[0])for v in line.split(" ")[1:]]

        # get coords
        poly_coords = []
        for v in vertexs:
            coords_line = obj_lines[v - 1 + vertex_line_start_idx]
            poly_coords.append([float(c)
                               for c in coords_line.split(" ")[1:]] + [1.0])

        # reduce polygon into triangles
        tri_coords = []
        for v_idx in range(1, len(poly_coords) - 1):
            tri_coords.append(
                [poly_coords[0], poly_coords[v_idx], poly_coords[v_idx + 1]])
        obj_tri_faces = np.vstack((obj_tri_faces, tri_coords))

    # coordinate of obj: z pointing up, x to right,  y to in scree
    # coordinate of program: z points in screen, x to right, y to up
    # so obj needs to be rotated -90 degrees around x axis
    #   and mirror around x-z plane

    # obj_tri_faces = rotate_mesh(obj_tri_faces, -np.pi/2, 0, 0)
    return obj_tri_faces


# test_nd_array = np.array([
#     [[1, 2, 3, 4],
#      [1, 2, 3, 4],
#      [1, 2, 3, 4]],
#     [[1, 2, 3, 4],
#      [1, 2, 3, 4],
#      [1, 2, 3, 4]]
# ])

# print(test_nd_array)


# def animate_by_matplotlib(
#         image_frame_func,
#         frames_index_array,
#         frames_pers_second):

#     import matplotlib.pyplot as plt
#     import time

#     plt.ion()
#     figure, ax = plt.subplots()

#     for frame_idx in frames_index_array:
#         plt.imshow(image_frame_func(frame_idx), cmap='gray')
#         figure.canvas.draw()
#         figure.canvas.flush_events()
#         time.sleep(1/frames_pers_second)

#     # The above code is too slow,
#     # so I will use matplotlib provided animation function

def animate_by_opencv(image_frame_func, frames_index_array, fps=30):

    # get the size of first frame to determine the size of the window
    height, width = image_frame_func(frames_index_array[0]).shape
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    for frame_idx in frames_index_array:

        pixel_matrix = image_frame_func(frame_idx)
        resized_pixel_matrix = cv2.resize(
            pixel_matrix, (width*5, height*5), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("image", resized_pixel_matrix)
        cv2.waitKey(int(1000/fps))

    cv2.destroyAllWindows()


def draw_line(srcBuf, row1, col1, row2, col2, colorDepth=1):
    """
    draw a 2d line. Color all of the pixels on the line black
    TODO: find a better way to draw lines
    """

    num_points = int(max(abs(col2 - col1), abs(row2 - row1))) + 1
    r_values = np.linspace(int(row1), int(row2), num_points, dtype=int)
    c_values = np.linspace(int(col1), int(col2), num_points, dtype=int)

    newSrcBuf = np.copy(srcBuf)
    newSrcBuf[r_values, c_values] = colorDepth
    newSrcBuf[newSrcBuf < 0] = 0
    newSrcBuf[newSrcBuf > 1] = 1

    return newSrcBuf


def draw_triangle_filled(srcBuf, triVertexs, colorDepth, colorDepth_func=None):
    """
    fills a face of triangle with the specified corlor
    colorDepth func is a function that takes in the coords of the pixel
        colorDepth_func(np.array[row],np.array[col]) -> np.array[colorDepth]

    https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    """

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

    if colorDepth_func is None:
        colorDepth_func = \
            lambda rows, cols: np.array(
                [colorDepth] * max(len(rows), len(cols)))

    # get the coords
    min_row = int(np.min(triVertexs[..., 0]))
    max_row = int(np.max(triVertexs[..., 0]))
    min_col = int(np.min(triVertexs[..., 1]))
    max_col = int(np.max(triVertexs[..., 1]))
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

        return srcBuf

    mask = coords_arr[(w_row >= 0) & (w_col >= 0) & (w_row + w_col <= 1)]
    mask_rows = mask[..., 0]
    mask_cols = mask[..., 1]

    # shading the mask
    colorDepth_arr = colorDepth_func(mask_rows, mask_cols)
    newSrcBuf = np.copy(srcBuf)
    newSrcBuf[mask_rows, mask_cols] = colorDepth_arr

    return newSrcBuf


def draw_triangle_frame(srcBuf, triVertexs, colorDepth=1):
    """
    takes in the screen buffer (2d ndarry),
    and a matrix[vertex_no][row, col]

    draws the triangle on the screenwith lines interconnecting it
    """

    newSrcBuf = srcBuf
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[1, 0], triVertexs[1, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[1, 0], triVertexs[1, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    return newSrcBuf


def coords_3d_to_screen_coords(coords, srcWidth, srcHeight):
    """
    coords 3d is a matrix[vertex_no][x, y, z, w]
        where x andh ystarts from -1 and ends at 1
        the origin is at (0,0)
    screen cords is a matrix[vertex_no][row, col]
        where row and col starts from 0 and ends at screenHeight and screenWidth
        the origin is at (0,0)    
    the transformation formula is as follows:
        row = (1 - y/ w)   * screenHeight / 2
        col = (1 + x/ w)   * screenWidth / 2
        the z and w is cut off.
    """

    # TODO：
    # Can we do the transformation in matrix form?

    newCoords = np.copy(coords)
    newCoords[..., 0] = (1 + coords[..., 1] /
                         coords[..., 3]) * (screenHeight-1) / 2
    newCoords[..., 1] = (1 - coords[..., 0] /
                         coords[..., 3]) * (srcWidth-1) / 2
    newCoords = newCoords[..., :2]
    newCoords = newCoords.astype(int)

    return newCoords


# rotate
def rotate(vec, x, y, z):
    """
    x y z means angle of rotation around around x y z axis in rad

    rotate a 3d vector, [x,y,z] or [x,y,z,w] 
    around a given axis, anticlockewise, for given degrees

    for example.
    rotate(vec, 0, 1/2 * np.pi, 0) rotates the vector around y axis
        anticlockwise for 90 degrees

    https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
    """

    # rotate_x_matrix = np.array(
    #     [[1, 0, 0, 0],
    #      [0, c(x), -s(x), 0],
    #      [0, s(x), c(x), 0],
    #      [0, 0, 0, 1]]
    # )

    # rotate_y_matrix = np.array(
    #     [[c(y), 0, s(y), 0],
    #      [0, 1, 0, 0],
    #      [-s(y), 0, c(y), 0],
    #      [0, 0, 0, 1]]
    # )

    # rotate_z_matrix = np.array(
    #     [[c(z), -s(z), 0, 0],
    #      [s(z), c(z), 0, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]]
    # )

    # rotation_matrix = rotate_z_matrix @ rotate_y_matrix @ rotate_x_matrix

    rotation_matrix = np.array(
        [[c(y)*c(z), c(z)*s(x)*s(y) - c(x)*s(z), s(x)*s(z) + c(x)*c(z)*s(y), 0],
         [c(y)*s(z), c(x)*c(z) + s(x)*s(y) * s(z), c(x)*s(y)*s(z) - c(z)*s(x), 0],
         [-s(y), c(y)*s(x), c(x)*c(y), 0],
         [0, 0, 0, 1]]
    )

    if vec.shape[0] == 3:
        return (rotation_matrix @ np.hstack((vec, [1])))[:3]
    else:
        return rotation_matrix @ vec


# rotate the mesh in 3d


def rotate_mesh(mesh, x_rad, y_rad, z_rad):
    """
    using matrix multiplication to rotate the mesh
    """

    rotated_mesh = np.zeros_like(mesh)
    for i in range(len(mesh)):
        for j in range(3):
            rotated_mesh[i, j] = rotate(mesh[i, j], x_rad, y_rad, z_rad)

    return rotated_mesh


# project the mesh into 2d screen coords
def project_onto_screen(mesh, screenWidth, screenHeight):

    projection_matrix = np.matrix(
        [[aspectRatio * 1/np.tan(fov/2), 0, 0, 0],
         [0, 1/np.tan(fov/2), 0, 0],
         [0, 0, far/(far - near), 1],
         [0, 0, -near * far/(far - near), 0]]
    )

    projected_2d_mesh = np.zeros_like(mesh)
    for i in range(len(mesh)):
        projected_2d_mesh[i] = mesh[i] @ projection_matrix
        projected_2d_mesh[i, ..., :2] = coords_3d_to_screen_coords(
            projected_2d_mesh[i], screenWidth, screenHeight)
    projected_2d_mesh = projected_2d_mesh[..., :2]
    return projected_2d_mesh

# calculate normal of every face of mesh


def calculate_normal(mesh):
    """
    input a mesh in 3d space in format of array[n][3][x,y,z,w]
    output an array of 3d vectors in dicating the normal of each face   
        in format of array[n][x,y,z]
    """

    normal_array = np.zeros((len(mesh), 3))
    for idx in range(0, len(mesh)):
        normal_array[idx] = normalize(np.cross(
            mesh[idx][1][:3] - mesh[idx][0][:3],
            mesh[idx][2][:3] - mesh[idx][0][:3]
        ))

    return normal_array

# clip


def clip(mesh,
         normal_array,
         camera_vec=np.array([0, 0, 0]),
         camera_posvec=np.array([0, 0, 0, 1]),
         camera_fov=90.0 / 360 * 2 * np.pi
         ):
    """
    parameters:
        - 3d mesh in format of mesh[n][3][x,y,z,w]
        - normal in for format of normal[n][x,y,z]
        - camera in format of camera[x,y,z]
    returns the clipped mesh in format of mesh[n][3][x,y,z,w]

    clips the mesh into triangles that are visible to the camera
    """

    clipped_mesh = np.empty((0, 3, 4))

    # simply normal dot camera will not work since we didn't take fov into account
    # and also will not work for camera not at origin.

    num_add = 0

    for idx in range(0, len(mesh)):
        center_coords = np.sum(mesh[idx], axis=0) / 3
        if normal_array[idx].dot((center_coords - camera_posvec)[:3]) < 0:
            num_add += 1
            clipped_mesh = np.vstack((clipped_mesh, [mesh[idx]]))

    return clipped_mesh

# # illuminating


def illuminating(normal_arr, light_vec=np.array([0, -1, 0])):
    """
    take in an array of normal and a light vector indicating the direction of light
    output an arrry of colordepth, corresponding to every triangle
    """

    return np.clip(normal_arr.dot(light_vec) + 0.5, 0, 1)


# normalise
def normalize(vec):
    return vec / np.linalg.norm(vec)

# animate


def rotation_animation(mesh, frame, frame_idx, fps=30):
    """
    takes a mesh objected, and output the screen projection 
    of it rotating on all 3 axis in different rates 
    """

    result_mesh = np.   copy(mesh)

    # rotate the mesh
    # calculate the angle based on frame
    x_rad = frame_idx / fps * 0.4 * np.pi
    z_rad = 0
    y_rad = frame_idx / fps * 0.3 * np.pi
    # z_rad = frame / fps * 0.8 * np.pi
    # z_rad = frame / 100 * 2 * np.pi
    result_mesh = rotate_mesh(result_mesh, x_rad, y_rad, z_rad)

    # distance it away from screen
    result_mesh[..., 2] += 20

    # calculate the normal of the mesh and clip it
    normal_arr = calculate_normal(result_mesh)
    result_mesh = clip(result_mesh, normal_array=normal_arr,
                       camera_vec=np.array([0, 0, 1]))

    # calculate the color depth of the mesh
    color_depth_arr = illuminating(calculate_normal(
        result_mesh), np.array([0, -1, 0]))

    # project onto 2d space
    projected_result_mesh = project_onto_screen(
        result_mesh, screenWidth, screenHeight)

    # draws on screen
    for i in range(len(projected_result_mesh)):
        frame = draw_triangle_filled(frame,
                                     projected_result_mesh[i],
                                     colorDepth=color_depth_arr[i])
        # frame = draw_triangle_frame(frame,
        #                                    projected_result_mesh[i],
        #                                    colorDepth=color_depth_arr[i])

        # TODO:
        # Draw twice here because the side of triangle will gitch
        # Fix the glitch.
        # Adding the line heavily reduces the performance.Fix ASAP!

    return frame


def main():
    origin_mesh = load_obj("test/teapot.obj")
    print(origin_mesh.shape)

    ## -------- Multithread Rendering Test -------- ##

    # frame_buffer = []

    # time1 = time.time()
    # with cf.ThreadPoolExecutor(max_workers=8) as executor:
    #     # Submit the tasks to the executor
    #     futures = [executor.submit(
    #         rotation_animation, origin_mesh, frame, i) for i in range(10)]

    #     # Wait for all tasks to complete
    #     cf.wait(futures)

    #     # Retrieve the results from completed tasks
    #     frame_buffer = [future.result() for future in futures]

    # time2 = time.time()

    # print(time2 - time1)

    # animate_by_opencv(lambda x: frame_buffer[x], range(0, 100), 20)

    # time3 = time.time()
    # print(time3-time2)

    ## -------- Single Threaad Rendering -------- ##

    animate_by_opencv(lambda x: rotation_animation(
        origin_mesh, frame, x, fps=30), range(1, 500), 30)


# if __name__ == "__main__":
#     exit(main())


"""
# TODO:
1. Phys Lab
2. Phys Notes
3. Profiler and read the fucking books
4. Maths Notes and Maths Thanksgiving Problem
5. Chem Course recordings
"""
