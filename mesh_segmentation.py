import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

if __name__=="__main__":

    fname = "./data/holding_eggs_under_arms.obj"   
    bunny_mesh = o3d.io.read_triangle_mesh(fname)

    vertices_o3d = bunny_mesh.vertices
    faces_o3d = bunny_mesh.triangles
    vertices = np.asarray(vertices_o3d)
    faces = np.asarray(faces_o3d)
    print("vertices shape: ", vertices.shape)
    print("faces shape: ", faces.shape)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d.visualization.draw_geometries([new_mesh])

    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]
    grouped = np.zeros(num_faces)
    for i in range(num_faces):
        if grouped[i] == 1:
            continue
    
