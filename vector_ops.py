import numpy as np
import variables
from maths_utils import normalize


def enlarge_vec(vec, factor):
    """
    resize a vector by a scale factor
    """

    return vec * factor


def rotate_vec(vec, x, y, z):
    """
    x y z means angle of rotation around around x y z axis in rad

    rotate a 3d vector, [x,y,z]]
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
    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)
    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)
    # rotation_matrix = np.array(
    #     [[c(y)*c(z), c(z)*s(x)*s(y) - c(x)*s(z), s(x)*s(z) + c(x)*c(z)*s(y)],
    #      [c(y)*s(z), c(x)*c(z) + s(x)*s(y) * s(z), c(x)*s(y)*s(z) - c(z)*s(x)],
    #      [-s(y), c(y)*s(x), c(x)*c(y)]]
    # )
    rotation_matrix = np.array(
        [[cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy],
         [cy*sz, cx*cz + sx*sy * sz, cx*sy*sz - cz*sx],
         [-sy, cy*sx, cx*cy]]
    )

    return rotation_matrix @ vec


def viewFromCamera_vec(vec, cameraPosVec, cameraUpVec, cameraDirectionVec):
    """
    https://www.youtube.com/watch?v=HXSuNxpCzdM&t=1567s
    """

    va = normalize(vec - cameraPosVec)
    vc = normalize(cameraUpVec - (cameraUpVec @ va) * va)
    vb = np.cross(vc, va)

    # print(cameraPosVec)
    newVec = np.hstack((vec, np.ones(1)))

    view_matrix = np.array(
        [
            [vb[0], vb[1], vb[2], 0],
            [vc[0], vc[1], vc[2], 0],
            [va[0], va[1], va[2], 0],
            [cameraPosVec[0], cameraPosVec[1], cameraPosVec[2], 1]
        ]
    )

    view_matrix = np.linalg.inv(view_matrix)

    newVec = view_matrix @  newVec

    newVec = newVec[:3] / newVec[3]

    return newVec


def project2screen_vec(vec, scrHeight, scrWidth):
    """
    This function takes a 3d vector of form [x,y,z] and make it 
    become [row, col, z], where z is not changed.
    """
    col = int((vec[0] + 1) / 2 * (scrWidth - 1))
    row = int((1 - (vec[1] + 1) / 2) * (scrHeight - 1))
    return np.array([row, col, vec[2]])


def project2viewCone_vec(vec, aspectRatio, fov, far, near):

    a = 1/np.tan(fov/2)
    b = far/(far - near)

    projection_matrix = np.array(
        [[aspectRatio * a, 0, 0, 0],
         [0, a, 0, 0],
         [0, 0, b, 1],
         [0, 0, -near * b, 0]]
    )

    # make vec homegenous
    vec = np.hstack((vec, np.ones(1)))

    vec = vec @ projection_matrix
    vec = vec[:3] / vec[3]
    return vec

# def inverseCam_vec(vec):


def apply_vecops_to_mesh(mesh, operation_func, *args, **kwargs):
    """
    Apply a vector operation function to each vector in the mesh.

    Parameters:
    - mesh: 3D mesh as a NumPy array
    - operation_func: Function to apply to each vector in the mesh
    - *args, **kwargs: Additional arguments to pass to the operation function

    Returns:
    - Resulting mesh after applying the operation
    """

    result_mesh = np.zeros((np.shape(mesh)[0], 3, 3))
    for i in range(np.shape(mesh)[0]):
        for j in range(3):
            result_mesh[i, j] = operation_func(mesh[i, j], *args, **kwargs)
    return result_mesh


# def apply_vecops_to_mesh_parallel(mesh, operation_func, *args, **kwargs):
#     """
#     Apply a vector operation function to each vector in the mesh.
#     (use parallel processing)
#     """

#     def parallel_operation(i, mesh, operation_func, args, kwargs):
#         result_row = np.zeros((3, 3))
#         for j in range(3):
#             result_row[j] = operation_func(mesh[i, j], *args, **kwargs)
#         return result_row

#     result_mesh = Parallel(n_jobs=PARALLEL_CORES)(delayed(parallel_operation)
#                                                   (i, mesh, operation_func, args, kwargs) for i in range(np.shape(mesh)[0]))
#     return np.array(result_mesh)
