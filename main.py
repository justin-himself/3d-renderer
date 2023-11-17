import concurrent.futures as cf
import numpy as np
from numpy import cos as c, sin as s
import cv2
import time
from vector_ops import *
from screen_ops import *
from mesh_ops import *
from obj_ops import *
from variables import *
from control_ops import *

# animate_by_maplotlib(lambda x: np.random.random((200, 300)), range(1, 100), 30)
s

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


def animate_by_opencv(image_frame_func, frames_index_array, fps=30, input_callback=None, timing=False):

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
        if input_callback is not None:
            input_callback(key)

        if timing:
            t2 = time.time()
            print("frame: ", frame_idx, "time: ", t2 - t1)

    cv2.destroyAllWindows()


# clip

# # illuminating


def illuminating(normal_arr, light_vec=np.array([0, -1, 0])):
    """
    take in an array of normal and a light vector indicating the direction of light
    output an arrry of colordepth, corresponding to every triangle
    """

    return np.clip(normal_arr.dot(light_vec) + 0.5, 0, 1)


def rotation_animation(mesh, frame_idx, fps=30):
    """
    takes a mesh objected, and output the screen projection 
    of it rotating on all 3 axis in different rates 
    """

    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
    result_mesh = np.copy(mesh)

    # rotate the mesh
    x_rad = y_rad = z_rad = 0
    x_rad = frame_idx / fps * 0.3 * np.pi
    y_rad = frame_idx / fps * 0.4 * np.pi
    # z_rad = frame_idx / fps * 0.8 * np.pi
    result_mesh = apply_vecops_to_mesh(
        result_mesh, rotate_vec, x_rad, y_rad, z_rad)

    # scale it down and distant away
    # result_mesh = apply_vecops_to_mesh(
    #     result_mesh, enlarge_vec, 0.3)
    result_mesh[..., 2] += 4

    # # do camera transformation
    # result_mesh = apply_vecops_to_mesh(
    #     result_mesh, viewFromCamera_vec, controllable_vars["camera_pos_vec"],
    #     controllable_vars["camera_up_vec"], controllable_vars["camera_direction_vec"])

    # project to view cone
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2viewCone_vec, ASPECT_RATIO, FOV, FAR, NEAR)

    # print(result_mesh)
    # illuminating
    illumanceArr = illuminating(calculate_normal(
        result_mesh), np.array([0, 1, 0]))

    # do normal clipping
    result_mesh = normal_clip(result_mesh, normal_array=calculate_normal(result_mesh),
                              cameraDir_vec=controllable_vars["camera_direction_vec"])

    # project to 2d screen
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2screen_vec, SCREEN_HEIGHT, SCREEN_WIDTH)

    # clip all the triangles against the screen edge
    result_mesh = clip_against_screen_edge(
        result_mesh, SCREEN_HEIGHT, SCREEN_WIDTH)

    def draw_frame(frame, projected_mesh, color_depth_arr):
        depthBuffer = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH)) * 1e7
        for i in range(len(projected_mesh)):
            draw_triangle(frame, projected_mesh[i], depthBuffer,
                          draw_triangle_filled, colorDepth=color_depth_arr[i])
            # draw_triangle(frame, projected_mesh[i], depthBuffer,
            #               draw_triangle_wireframe, colorDepth=1)
            # draw_triangle(frame, projected_mesh[i], depthBuffer,
            #               draw_triangle_filled, colorDepth=1)
    draw_frame(frame, result_mesh, illumanceArr)
    return frame


def main():
    origin_mesh = load_objfile("models/cube_6.obj")
    # origin_mesh = load_objfile("models/rectgrid_16.obj")
    # origin_mesh = load_objfile("models/teapot_158.obj")
    # origin_mesh = load_objfile("models/axis_140.obj")
    animate_by_opencv(lambda x: rotation_animation(
        origin_mesh, x, fps=30), range(1, 500), 30, input_callback=user_control, timing=False)


if __name__ == "__main__":
    exit(main())
