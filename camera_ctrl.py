from variables import *
from vector_ops import *

#####################
# UTILITY FUNCTIONS #
#####################


def get_cam_dir_forward():
    cam_dir_forward = controllable_vars['camera_direction_vec']
    cam_dir_forward[1] = 0
    cam_dir_forward = normalize(cam_dir_forward)
    return cam_dir_forward


def get_cam_dir_right():
    cam_dir_forward = get_cam_dir_forward()
    cam_dir_right = normalize(
        np.cross(controllable_vars['camera_up_vec'], cam_dir_forward))
    return cam_dir_right

###########################
# CAMERA MOVEMENT CONTROL #
###########################


def cam_move_forward():
    cam_dir_forward = get_cam_dir_forward()
    controllable_vars['camera_pos_vec'] += cam_dir_forward * \
        controllable_vars['camera_move_speed']


def cam_move_backward():
    cam_dir_forward = get_cam_dir_forward()
    controllable_vars['camera_pos_vec'] -= cam_dir_forward * \
        controllable_vars['camera_move_speed']


def cam_move_left():
    cam_dir_right = get_cam_dir_right()
    controllable_vars['camera_pos_vec'] -= cam_dir_right * \
        controllable_vars['camera_move_speed']


def cam_move_right():
    cam_dir_right = get_cam_dir_right()
    controllable_vars['camera_pos_vec'] += cam_dir_right * \
        controllable_vars['camera_move_speed']
