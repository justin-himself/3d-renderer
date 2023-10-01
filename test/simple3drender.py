import numpy as np
import cv2


# https://www.youtube.com/watch?v=ih20l3pJoeU


def load_obj(file_path):
    """
    take object file and output a matrix descriping 3d mesh,
    in the format of 

        matrix[tri_no][vertex_no][x, y, z, w]

    Where always equals to 1. (Homogeneous Coordinates)
    Plogon faces will be converted into component triangles.
    """

    with open(file_path) as obj_file:
        obj_lines = obj_file.readlines()

    # rules:
        # v x y z       - vertex with coord x,y,z
        # f a b c...    - face with vertex indexed a b c

    # find starting idx of vertexs
    vertex_line_base_idx = 0
    for idx, line in enumerate(obj_lines):
        if line[:2] == "v ":
            vertex_line_start_idx = idx
            break

    # load object into a matrix like
    # matrix[tri_no][vertex_no][x, y, z, w]
    obj_tri_faces = np.empty((0, 3, 4))
    for line in obj_lines:
        if line[:2] != "f ":
            continue
        vertexs = [int(v.split("/")[0])for v in line.split(" ")[1:]]

        # get coords
        poly_coords = []
        for v in vertexs:
            coords_line = obj_lines[v - 1 + vertex_line_start_idx]
            poly_coords.append([float(c)
                               for c in coords_line.split(" ")[1:]] + [1.0])

        # reduce polygon into triangles
        tri_coords = []
        for v_idx in range(1, len(poly_coords) - 1):
            tri_coords.append(
                [poly_coords[0], poly_coords[v_idx], poly_coords[v_idx + 1]])
        obj_tri_faces = np.vstack((obj_tri_faces, tri_coords))

    return obj_tri_faces


# test_nd_array = np.array([
#     [[1, 2, 3, 4],
#      [1, 2, 3, 4],
#      [1, 2, 3, 4]],
#     [[1, 2, 3, 4],
#      [1, 2, 3, 4],
#      [1, 2, 3, 4]]
# ])

# print(test_nd_array)


# def animate_by_maplotlib(
#         image_frame_func,
#         frames_index_array,
#         frames_pers_second):

#     plt.ion()
#     figure, ax = plt.subplots()

#     for frame_idx in frames_index_array:
#         plt.imshow(image_frame_func(frame_idx), cmap='gray')
#         figure.canvas.draw()
#         figure.canvas.flush_events()
#         time.sleep(1/frames_pers_second)
        
#     # The above code is too slow, 
#     # so I will use matplotlib provided animation function
    
def animate_by_opencv(
        image_frame_func,
        frames_index_array,
        frames_pers_second):
    
    # get the size of first frame to determine the size of the window
    height, width = image_frame_func(frames_index_array[0]).shape
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    
 
    for frame_idx in frames_index_array:
        pixel_matrix = image_frame_func(frame_idx)
        resized_pixel_matrix = cv2.resize(pixel_matrix, (width*5, height*5), interpolation=cv2.INTER_NEAREST)
    
        cv2.imshow("image", resized_pixel_matrix)
        cv2.waitKey(int(1000/frames_pers_second))
    
    cv2.destroyAllWindows()
    



# animate_by_maplotlib(lambda x: np.random.random((200, 300)), range(1, 100), 30)
screenHeight = 100
screenWidth = 100
aspectRatio = screenHeight / screenWidth
fov = 90.0 / 360 * 2 * np.pi
far = 1000
near = 1

screenBuffer = np.ones((screenHeight, screenWidth))
screenBuffer[0, 0] = 0

projection_matrix = np.matrix(
    [[aspectRatio * 1/np.tan(fov/2), 0, 0, 0],
     [0, 1/np.tan(fov/2), 0, 0],
     [0, 0, far/(far - near), 1],
     [0, 0, -near * far/(far - near), 0]]
)


def draw_line(srcBuf, row1, col1, row2, col2, colorDepth=1):
    """
    draw a 2d line. Color all of the pixels on the line black
    TODO: find a better way to draw lines
    """

    num_points = int(max(abs(col2 - col1), abs(row2 - row1))) + 1
    r_values = np.linspace(int(row1), int(row2), num_points, dtype=int)
    c_values = np.linspace(int(col1), int(col2), num_points, dtype=int)

    newSrcBuf = np.copy(srcBuf)
    newSrcBuf[r_values, c_values] -= colorDepth
    newSrcBuf[newSrcBuf < 0] = 0
    newSrcBuf[newSrcBuf > 1] = 1

    return newSrcBuf


def draw_triangle(srcBuf, triVertexs, colorDepth=1):
    """
    takes in the screen buffer (2d ndarry),
    and a matrix[vertex_no][row, col]

    draws the triangle on the screenwith lines interconnecting it
    """

    newSrcBuf = srcBuf
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[1, 0], triVertexs[1, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[0, 0], triVertexs[0, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    newSrcBuf = draw_line(
        newSrcBuf, triVertexs[1, 0], triVertexs[1, 1], triVertexs[2, 0], triVertexs[2, 1], colorDepth)
    return newSrcBuf

def coords_3d_to_screen_coords(coords, srcWidth, srcHeight):
    """
    coords 3d is a matrix[vertex_no][x, y, z, w]
        where x andh ystarts from -1 and ends at 1
        the origin is at (0,0)
    screen cords is a matrix[vertex_no][row, col]
        where row and col starts from 0 and ends at screenHeight and screenWidth
        the origin is at (0,0)    
    the transformation formula is as follows:
        row = (1 - y/ w)   * screenHeight / 2
        col = (1 + x/ w)   * screenWidth / 2
        the z and w is cut off.
    """

    newCoords = np.copy(coords)
    newCoords[..., 0] = (1 + coords[..., 0]/coords[..., 3]) * (screenHeight-1) / 2
    newCoords[..., 1] = (1 - coords[..., 1]/ coords[..., 3]) * (srcWidth-1) / 2
    newCoords = newCoords[..., :2]
    newCoords = newCoords.astype(int)

    return newCoords




# rotate the mesh in 3d
def rotation(mesh, x_rad, y_rad, z_rad):
    """
    using matrix multiplication to rotate the mesh
    """

    rotate_x_matrix = np.array(
        [[1, 0, 0, 0],
        [0, np.cos(x_rad), np.sin(x_rad), 0],
        [0, -np.sin(x_rad), np.cos(x_rad), 0],
        [0, 0, 0, 1]]
    )

    rotate_y_matrix = np.array(
        [[np.cos(y_rad), 0, -np.sin(y_rad), 0],
        [0, 1, 0, 0],
        [np.sin(y_rad), 0, np.cos(y_rad), 0],
        [0, 0, 0, 1]]
    )

    rotate_z_matrix = np.array(
        [[np.cos(z_rad), np.sin(z_rad), 0, 0],
        [-np.sin(z_rad), np.cos(z_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    rotation_matrix = rotate_x_matrix @ rotate_y_matrix @ rotate_z_matrix

    rotated_mesh = np.zeros_like(mesh)
    for i in range(len(mesh)):
        rotated_mesh[i] = mesh[i] @ rotation_matrix

    return rotated_mesh
        
        

# project the mesh into 2d screen coords
def project_onto_screen(mesh, screenWidth, screenHeight):
    projected_2d_mesh = np.zeros_like(mesh)
    for i in range(len(mesh)):
        projected_2d_mesh[i] = mesh[i] @ projection_matrix    
        projected_2d_mesh[i, ..., :2] = coords_3d_to_screen_coords(
            projected_2d_mesh[i], screenWidth, screenHeight)
    projected_2d_mesh = projected_2d_mesh[..., :2]
    return projected_2d_mesh

# animate
def rotation_animation(mesh, screenBuffer, frame):
    """
    takes a mesh objected, and output the screen projection 
    of it rotating on all 3 axis in different rates 
    """

    result_mesh = np.copy(mesh)

    # distance it away from screen

    # rotate the mesh
    # calculate the angle based on frame
    x_rad = frame / 30 * 0.4 * np.pi
    y_rad = frame / 30 * 0.3 * np.pi
    z_rad = 0
    # z_rad = frame / 100 * 2 * np.pi
    result_mesh = rotation(result_mesh, x_rad, y_rad, z_rad)


    result_mesh[..., 2] += 5
    # project onto 2d space
    projected_result_mesh = project_onto_screen(result_mesh, screenWidth, screenHeight)

    # draws on screen
    for i in projected_result_mesh:
        screenBuffer = draw_triangle(screenBuffer, i)

    return screenBuffer

    


origin_mesh = load_obj("test/cube.obj")


animate_by_opencv(lambda x: rotation_animation(origin_mesh, screenBuffer, x), range(1, 10000), 30)
