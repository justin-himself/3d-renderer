import numpy as np
from maths_utils import normalize

#####################
# UTILITY FUNCTIONS #
#####################


def get_cam_dir_forward(controllable_vars):

    cam_dir_forward = controllable_vars.camera_direction_vec
    cam_dir_forward[1] = 0
    cam_dir_forward = normalize(cam_dir_forward)
    return cam_dir_forward


def get_cam_dir_right(controllable_vars):

    cam_dir_forward = get_cam_dir_forward(controllable_vars)
    cam_dir_right = normalize(
        np.cross(controllable_vars.camera_up_vec, cam_dir_forward))
    return cam_dir_right

###########################
# CAMERA MOVEMENT CONTROL #
###########################


def cam_move_forward(controllable_vars):

    cam_dir_forward = get_cam_dir_forward(controllable_vars)
    controllable_vars.camera_pos_vec += cam_dir_forward * \
        controllable_vars.camera_move_speed


def cam_move_backward(controllable_vars):

    cam_dir_forward = get_cam_dir_forward(controllable_vars)
    controllable_vars.camera_pos_vec -= cam_dir_forward * \
        controllable_vars.camera_move_speed


def cam_move_left(controllable_vars):

    cam_dir_right = get_cam_dir_right(controllable_vars)
    controllable_vars.camera_pos_vec -= cam_dir_right * \
        controllable_vars.camera_move_speed


def cam_move_right(controllable_vars):

    cam_dir_right = get_cam_dir_right(controllable_vars)
    controllable_vars.camera_pos_vec += cam_dir_right * \
        controllable_vars.camera_move_speed


def cam_move_up(controllable_vars):

    controllable_vars.camera_pos_vec[1] += controllable_vars.camera_move_speed


def cam_move_down(controllable_vars):

    controllable_vars.camera_pos_vec[1] -= controllable_vars.camera_move_speed
