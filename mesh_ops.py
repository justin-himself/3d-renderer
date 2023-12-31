import numpy as np
from maths_utils import normalize, is_point_in_plane, line_plane_intersect, polygon_to_triangles
from vector_ops import rotate_vec


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


def illuminating(normal_arr, light_vec=np.array([0, -1, 0])):
    """
    take in an array of normal and a light vector indicating the direction of light
    output an arrry of colordepth, corresponding to every triangle
    """

    return np.clip(normal_arr.dot(light_vec) + 0.5, 0, 1)


def normal_clip(mesh, normal_array, cameraDir_vec):
    """
    clips the mesh into triangles that are visible to the camera,
    acoording to the camera look direction

    returns an index array of the visible triangles
    """

    # clipped_mesh = np.empty((0, 3, 3))
    clipped_mesh_idx = np.empty((0), dtype=int)

    # simply normal dot camera will not work since we didn't take fov into account
    # and also will not work for camera not at origin.

    num_add = 0

    l = np.shape(mesh)[0]
    for idx in range(0, l):
        center_coords = np.sum(mesh[idx], axis=0) / 3
        if normal_array[idx].dot((cameraDir_vec)) < 0:
            num_add += 1
            # clipped_mesh = np.vstack((clipped_mesh, [mesh[idx]]))
            clipped_mesh_idx = np.append(clipped_mesh_idx, idx)

    return clipped_mesh_idx


def triangle_plane_clip(triVertexs, planePoint, planeNormal):
    """
    clips a triangle against a plane
    returns a list of triangles
    """

    vertexList = []
    outCnt = 0
    for triVex in triVertexs:
        if is_point_in_plane(triVex, planePoint, planeNormal):
            vertexList.append((triVex, "i"))
        else:
            vertexList.append((triVex, "o"))
            outCnt += 1

    if outCnt == 3:  # all 3 vertexes are out of one bound
        return np.zeros((0, 3, 3))
    elif outCnt == 0:  # all 3 vertexes are in bound
        return np.array([triVertexs])

    polygon_vextexs = []
    vertexList.append(vertexList[0])
    for idx, vex in enumerate(vertexList[:-1]):
        if vex[1] == "i":
            polygon_vextexs.append(vex[0])
            vexnext = vertexList[idx + 1]
            if vexnext[1] == "o":
                intersectCoord = line_plane_intersect(
                    vex[0], vexnext[0],
                    planePoint, planeNormal
                )
                polygon_vextexs.append(intersectCoord)
        else:
            vexnext = vertexList[idx + 1]
            if vexnext[1] == "i":
                intersectCoord = line_plane_intersect(
                    vex[0], vexnext[0],
                    planePoint, planeNormal
                )
                polygon_vextexs.append(intersectCoord)

    return polygon_to_triangles(np.array(polygon_vextexs))


def mesh_plane_clip(mesh, planePoint, planeNormal, illuminanceArr=None):
    """
    clips a mesh against a plane
    returns a list of triangles
    """
    resultMesh = np.zeros((0, 3, 3))
    resultIluArr = np.zeros((0))

    for idx, tri in enumerate(mesh):
        triList = triangle_plane_clip(tri, planePoint, planeNormal)
        resultMesh = np.vstack((resultMesh, triList))
        if illuminanceArr is not None:
            resultIluArr = np.append(
                resultIluArr, np.full((len(triList)), illuminanceArr[idx]))

    return resultMesh, resultIluArr


def clip_against_screen_edge(mesh, screenHeight, screenWidth, illuminanceArr=None):
    iluArr = illuminanceArr
    mesh, iluArr = mesh_plane_clip(mesh, np.array(
        [screenHeight - 1, 0, 0]), np.array([1, 0, 0]), iluArr)
    mesh, iluArr = mesh_plane_clip(mesh, np.array(
        [0, 0, 0]), np.array([-1, 0, 0]), iluArr)
    mesh, iluArr = mesh_plane_clip(mesh, np.array(
        [0, screenWidth - 1, 0]), np.array([0, 1, 0]), iluArr)
    mesh, iluArr = mesh_plane_clip(mesh, np.array(
        [0, 0, 0]), np.array([0, -1, 0]), iluArr)

    if illuminanceArr is None:
        return mesh
    return mesh, iluArr


def clip_behind_camera(mesh, cameraPos_vec, cameraDir_vec):
    """
    clips a mesh against a plane
    returns a list of triangles
    """
    return mesh_plane_clip(mesh, cameraPos_vec, -cameraDir_vec)


def clip_outside_viewcone(mesh, cameraPos_vec, cameraUp_vec, fov):
    # print(rotate_vec(
    #     np.array([0, 0, 1]), 0, -variables.FOV/2, 0))
    leftPlaneNormal = np.cross(rotate_vec(
        np.array([0, 0, 1]), 0, -variables.FOV/2, 0), cameraUp_vec)
    # print(leftPlaneNormal)
    mesh = mesh_plane_clip(mesh, cameraPos_vec,
                           leftPlaneNormal)
    # mesh = mesh_plane_clip(mesh, cameraPos_vec,
    #                        rightPlaneNormal)
    # mesh = mesh_plane_clip(mesh, cameraPos_vec,
    #                        topPlaneNormal)
    # mesh = mesh_plane_clip(mesh, cameraPos_vec,
    #                        bottomPlaneNormal)
    return mesh


def illuminance_clip(mesh, illuminanceArr, threshold=0.01):
    # 1 is brightest, 0 is darkest

    return mesh[illuminanceArr >= threshold]
