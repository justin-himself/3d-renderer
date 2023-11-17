import numpy as np
from vector_ops import *
from variables import *
from camera_ctrl import *

user_control_keymap = {
    # 0: cam_look_up,
    # 1: cam_look_down,
    # 2: cam_look_left,
    # 3: cam_look_right,
    ord('w'): cam_move_forward,
    ord('s'): cam_move_backward,
    ord('a'): cam_move_left,
    ord('d'): cam_move_right,
}


def user_control(key_press):

    if key_press == -1:
        return

    if key_press >= ord('A') and key_press <= ord('Z'):
        key_press = key_press + ord('a') - ord('A')

    if key_press in user_control_keymap:
        user_control_keymap[key_press]()
