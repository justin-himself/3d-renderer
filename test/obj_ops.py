import os
import re
import numpy as np
from mesh_ops import *
from vector_ops import *


def load_objfile(file_path):
    """
    take object file and output a matrix descriping 3d mesh,
    in the format of 

        matrix[tri_no][vertex_no][x, y, z]

    Where always equals to 1. (Homogeneous Coordinates)
    Plogon faces will be converted into component triangles.
    """

    with open(file_path) as obj_file:
        obj_lines = obj_file.readlines()

    # rules:
        # v x y z       - vertex with –oord x,y,z
        # f a b c...    - face with vertex indexed a b c

    # find starting idx of vertexs
    for idx, line in enumerate(obj_lines):
        if line[:2] == "v ":
            vertex_line_start_idx = idx
            break

    # load object into a matrix like
    # matrix[tri_no][vertex_no][x, y, z]
    mesh = np.empty((0, 3, 3))

    for line in obj_lines:
        if line[:2] != "f ":
            continue
        vertexs = [int(v.split("/")[0]) for v in line.split(" ")[1:]]

        # get coords
        poly_coords = []
        for v in vertexs:
            coords_line = obj_lines[v - 1 + vertex_line_start_idx]
            poly_coords.append([float(c)
                               for c in coords_line.split(" ")[1:]])

        # reduce polygon into triangles
        tri_coords = []
        for v_idx in range(1, len(poly_coords) - 1):
            tri_coords.append(
                [poly_coords[0], poly_coords[v_idx], poly_coords[v_idx + 1]])

        try:
            mesh = np.vstack((mesh, tri_coords))
        except:
            exit(0)

    # scale down the mesh
    scale_factor = 1 / np.max(get_boundary_box(mesh))
    mesh = apply_vecops_to_mesh(mesh, enlarge_vec, scale_factor)

    # coordinate of obj: z pointing up, x to right,  y to in scree
    # coordinate of program: z points in screen, x to right, y to up
    # so obj needs to be rotated -90 degrees around x axis
    #   and mirror around x-z plane

    # obj_tri_faces = rotate_mesh(obj_tri_faces, -np.pi/2, 0, 0)
    return mesh


def load_obj(folder_path, objname):
    """
    Find the obj file in the folder, in format like
    <objname>_<facecount>.obj

    And return an ordered list of mesh in order of accending facecount
    """

    # find all obj files in the folder
    obj_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".obj") and file.startswith(objname + "_"):
            obj_files.append(file)

    meshes = []
    facecounts = []
    for obj_file in obj_files:
        facecounts.append(int(re.search(r'\d+', obj_file).group()))
        meshes.append(load_objfile(folder_path + obj_file))

    meshes = [mesh for _, mesh in sorted(zip(facecounts, meshes))]
    return meshes
