import concurrent.futures as cf
import numpy as np
from numpy import cos as c, sin as s
import cv2
import time
from vector_ops import *
from screen_ops import *
from constants import *

# animate_by_maplotlib(lambda x: np.random.random((200, 300)), range(1, 100), 30)

frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))


def load_obj(file_path):
    """
    take object file and output a matrix descriping 3d mesh,
    in the format of 

        matrix[tri_no][vertex_no][x, y, z]

    Where always equals to 1. (Homogeneous Coordinates)
    Plogon faces will be converted into component triangles.
    """

    with open(file_path) as obj_file:
        obj_lines = obj_file.readlines()

    # rules:
        # v x y z       - vertex with –oord x,y,z
        # f a b c...    - face with vertex indexed a b c

    # find starting idx of vertexs
    for idx, line in enumerate(obj_lines):
        if line[:2] == "v ":
            vertex_line_start_idx = idx
            break

    # load object into a matrix like
    # matrix[tri_no][vertex_no][x, y, z]
    mesh = np.empty((0, 3, 3))

    for line in obj_lines:
        if line[:2] != "f ":
            continue
        vertexs = [int(v.split("/")[0]) for v in line.split(" ")[1:]]

        # get coords
        poly_coords = []
        for v in vertexs:
            coords_line = obj_lines[v - 1 + vertex_line_start_idx]
            poly_coords.append([float(c)
                               for c in coords_line.split(" ")[1:]])

        # reduce polygon into triangles
        tri_coords = []
        for v_idx in range(1, len(poly_coords) - 1):
            tri_coords.append(
                [poly_coords[0], poly_coords[v_idx], poly_coords[v_idx + 1]])

        mesh = np.vstack((mesh, tri_coords))

    # scale down the mesh
    scale_factor = 1 / np.max(get_bounding_box(mesh))
    mesh = apply_vecops_to_mesh(mesh, enlarge_vec, scale_factor)

    # coordinate of obj: z pointing up, x to right,  y to in scree
    # coordinate of program: z points in screen, x to right, y to up
    # so obj needs to be rotated -90 degrees around x axis
    #   and mirror around x-z plane

    # obj_tri_faces = rotate_mesh(obj_tri_faces, -np.pi/2, 0, 0)
    return mesh


def get_bounding_box(mesh):
    x_max = y_max = z_max = 0
    for tri in mesh:
        for vertex in tri:
            x_max = max(x_max, vertex[0])
            y_max = max(y_max, vertex[1])
            z_max = max(z_max, vertex[2])
    return [x_max, y_max, z_max]


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

def animate_by_opencv(image_frame_func, frames_index_array, fps=30, timing=False):

    # get the size of first frame to determine the size of the window
    height, width = image_frame_func(frames_index_array[0]).shape
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    for frame_idx in frames_index_array:

        if timing:
            t1 = time.time()

        pixel_matrix = image_frame_func(frame_idx)
        resized_pixel_matrix = cv2.resize(
            pixel_matrix, (width*5, height*5), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("image", resized_pixel_matrix)
        key = cv2.waitKey(int(1000/fps))

        if timing:
            t2 = time.time()
            print("frame: ", frame_idx, "time: ", t2 - t1)

    cv2.destroyAllWindows()


def calculate_normal(mesh):
    """
    input a mesh in 3d space in format of array[n][3][x,y,z]
    output an array of 3d vectors in dicating the normal of each face   
        in format of array[n][x,y,z]
    """
    l = np.shape(mesh)[0]
    normal_array = np.zeros((l, 3))
    for idx in range(0, l):
        normal_array[idx] = normalize(np.cross(
            mesh[idx][1] - mesh[idx][0],
            mesh[idx][2] - mesh[idx][0]
        ))

    return normal_array

# clip


def clip(mesh,
         normal_array,
         camera_vec=np.array([0, 0, 0]),
         camera_posvec=np.array([0, 0, 0]),
         camera_fov=90.0 / 360 * 2 * np.pi
         ):
    """
    parameters:
        - 3d mesh in format of mesh[n][3][x,y,z]
        - normal in for format of normal[n][x,y,z]
        - camera in format of camera[x,y,z]
    returns the clipped mesh in format of mesh[n][3][x,y,z]

    clips the mesh into triangles that are visible to the camera
    """

    clipped_mesh = np.empty((0, 3, 3))

    # simply normal dot camera will not work since we didn't take fov into account
    # and also will not work for camera not at origin.

    num_add = 0

    l = np.shape(mesh)[0]
    for idx in range(0, l):
        center_coords = np.sum(mesh[idx], axis=0) / 3
        if normal_array[idx].dot((center_coords - camera_posvec)) < 0:
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


def rotation_animation(mesh, frame, frame_idx, fps=30):
    """
    takes a mesh objected, and output the screen projection 
    of it rotating on all 3 axis in different rates 
    """

    result_mesh = np.copy(mesh)

    # rotate the mesh
    # calculate the angle based on frame
    x_rad = y_rad = z_rad = 0
    y_rad = frame_idx / fps * 0.4 * np.pi
    x_rad = frame_idx / fps * 0.3 * np.pi
    # z_rad = frame / fps * 0.8 * np.pi
    result_mesh = apply_vecops_to_mesh(
        result_mesh, rotate_vec, x_rad, y_rad, z_rad)

    # distance it away from screen
    result_mesh[..., 2] += 1.5

    # calculate the normal of the mesh and clip it
    normal_arr = calculate_normal(result_mesh)
    result_mesh = clip(result_mesh, normal_array=normal_arr,
                       camera_vec=np.array([0, 0, 1]))

    # calculate the color depth of the mesh
    color_depth_arr = illuminating(calculate_normal(
        result_mesh), np.array([0, 1, 0]))

    # do the projection
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2viewCone_vec, ASPECT_RATIO, FOV, FAR, NEAR)
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2screen_vec, SCREEN_HEIGHT, SCREEN_WIDTH)

    def draw_frame(frame, projected_mesh, color_depth_arr):
        depthBuffer = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH)) * 1e7
        for i in range(len(projected_mesh)):
            frame = draw_triangle(frame, projected_mesh[i], depthBuffer,
                                  draw_triangle_filled, colorDepth=color_depth_arr[i])
        return frame

    frame = draw_frame(frame, result_mesh, color_depth_arr)
    return frame


def main():
    origin_mesh = load_obj("test/teapot.obj")
    animate_by_opencv(lambda x: rotation_animation(
        origin_mesh, frame, x, fps=30), range(1, 500), 30, timing=True)


if __name__ == "__main__":
    exit(main())
