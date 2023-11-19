import numpy as np
import variables
from vector_ops import apply_vecops_to_mesh, rotate_vec, viewFromCamera_vec, project2viewCone_vec, project2screen_vec
from screen_ops import draw_triangle, draw_triangle_filled, draw_triangle_wireframe
from mesh_ops import illuminating, normal_clip, calculate_normal, clip_against_screen_edge


def rotation_animation(
        mesh,
        frame_idx,
        controllable_vars,
        draw_wireframe=True,
        draw_filled=False,
        do_normal_clip=True,
        screen_width=variables.SCREEN_WIDTH,
        screen_height=variables.SCREEN_HEIGHT):
    """
    takes a mesh objected, and output the screen projection 
    of it rotating on all 3 axis in different rates 
    """

    frame = np.zeros((screen_height, screen_width))
    result_mesh = mesh

    # rotate the mesh
    x_rad = y_rad = z_rad = 0
    x_rad = frame_idx * 0.01 * np.pi
    y_rad = frame_idx * 0.02 * np.pi
    # z_rad = frame_idx / fps * 0.8 * np.pi
    result_mesh = apply_vecops_to_mesh(
        result_mesh, rotate_vec, x_rad, y_rad, z_rad)

    # distant away from screen
    result_mesh[..., 2] += 2

    # # clip everyhting behind the camera
    # result_mesh, _ = clip_behind_camera(
    #     result_mesh, controllable_vars.camera_pos_vec, controllable_vars.camera_direction_vec)

    # # clip everyhing outside view cone
    # result_mesh = clip_outside_viewcone(
    #     result_mesh, controllable_vars.camera_pos_vec,  controllable_vars.camera_up_vec, variables.FOV)
    # illuminating
    illuminanceArr = illuminating(calculate_normal(
        result_mesh), np.array([0, 1, 0]))

    # do camera transformation
    result_mesh = apply_vecops_to_mesh(
        result_mesh, viewFromCamera_vec, controllable_vars.camera_pos_vec,
        controllable_vars.camera_up_vec, controllable_vars.camera_direction_vec)

    # project to view cone
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2viewCone_vec, variables.ASPECT_RATIO, variables.FOV, variables.FAR, variables.NEAR)

    # do normal clipping
    # normal clip increases performance, but will introduce errors (false clipping)
    # will be useful in preview since performance is important
    if do_normal_clip:
        clipped_mesh_idx = normal_clip(result_mesh, normal_array=calculate_normal(result_mesh),
                                       cameraDir_vec=controllable_vars.camera_direction_vec)
        result_mesh = result_mesh[clipped_mesh_idx]
        illuminanceArr = illuminanceArr[clipped_mesh_idx]

    # project to 2d screen
    result_mesh = apply_vecops_to_mesh(
        result_mesh, project2screen_vec, screen_height, screen_width)
    # clip all the triangles against the screen edge
    result_mesh, illuminanceArr = clip_against_screen_edge(
        result_mesh, screen_height, screen_width, illuminanceArr)

    def draw_frame(frame, projected_mesh, color_depth_arr):
        depthBuffer = np.ones((screen_height, screen_width)) * 1e7
        for i in range(len(projected_mesh)):
            if draw_wireframe:
                draw_triangle(frame, projected_mesh[i], depthBuffer,
                              draw_triangle_wireframe, colorDepth=1)
            if draw_filled:
                draw_triangle(frame, projected_mesh[i], depthBuffer,
                              draw_triangle_filled, colorDepth=color_depth_arr[i])
                draw_triangle(frame, projected_mesh[i], depthBuffer,
                              draw_triangle_wireframe, colorDepth=color_depth_arr[i])
    draw_frame(frame, result_mesh, illuminanceArr)

    return frame
