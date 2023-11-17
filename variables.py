import numpy as np

PARALLEL_CORES = 6

SCREEN_HEIGHT = 100
SCREEN_WIDTH = SCREEN_HEIGHT
ASPECT_RATIO = SCREEN_HEIGHT / SCREEN_WIDTH
FOV = 90.0 / 360 * 2 * np.pi
FAR = 1000
NEAR = 1


controllable_vars = {
    'camera_pos_vec': np.array([0, 0, 0], dtype=np.float64),
    'camera_up_vec': np.array([0, 1, 0], dtype=np.float64),
    'camera_direction_vec': np.array([0, 0, 1], dtype=np.float64),
    'camera_rotate_speed': 0.1,  # rad
    'camera_move_speed': 0.5,
}
