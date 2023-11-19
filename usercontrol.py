import numpy as np
from camera_ctrl import cam_move_backward, cam_move_down, cam_move_forward, cam_move_left, cam_move_right, cam_move_up


def quit_preview(controllable_vars):

    controllable_vars.preview_ends = True


user_control_keymap = {
    'q': cam_move_forward,
    'e': cam_move_backward,
    'a': cam_move_left,
    'd': cam_move_right,
    's': cam_move_up,
    'w': cam_move_down,
    'enter': quit_preview
}


class ControlableVars:
    def __init__(self):
        self.camera_pos_vec = np.array([0, 0, 0], dtype=np.float64)
        self.camera_up_vec = np.array([0, 1, 0], dtype=np.float64)
        self.camera_direction_vec = np.array([0, 0, 1], dtype=np.float64)
        self.camera_rotate_speed = 0.1  # rad
        self.camera_move_speed = 0.5
        self.preview_ends = False


def print_keymap():
    print("------ keymap -----")
    for key in user_control_keymap:
        print(f"{key}: {user_control_keymap[key].__name__}")
    print("-------------------")
