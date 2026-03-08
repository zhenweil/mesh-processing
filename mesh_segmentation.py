import vedo
import trimesh
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    fname = "/home/zhenweil/mesh-processing/data/holding_eggs_under_arms.obj"   
    mesh = trimesh.load(fname)
    vertices = mesh.vertices
    faces = mesh.faces

    new_mesh = vedo.Mesh([vertices, faces])
    vedo.show(new_mesh)
