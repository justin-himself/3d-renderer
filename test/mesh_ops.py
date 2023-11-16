import numpy as np
from maths_ops import *


def get_boundary_box(mesh):
    x_max = y_max = z_max = 0
    for tri in mesh:
        for vertex in tri:
            x_max = max(x_max, vertex[0])
            y_max = max(y_max, vertex[1])
            z_max = max(z_max, vertex[2])
    return [x_max, y_max, z_max]


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


def normal_clip(mesh,
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


def illuminance_clip(mesh, illuminanceArr, threshold=0.01):
    # 1 is brightest, 0 is darkest

    return mesh[illuminanceArr >= threshold]
