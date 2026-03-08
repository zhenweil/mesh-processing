import vedo
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class EZMesh:
    def __init__(self, fname):
        mesh = trimesh.load(fname)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.normals = mesh.face_normals
        self.num_vertices = len(self.vertices)
        self.num_faces = len(self.faces)
        self.vtx_to_vtx = [set() for _ in range(self.num_vertices)]
        self.vtx_to_face = [set() for _ in range(self.num_vertices)]
        self.edge_to_face = defaultdict(set)

        for i in range(self.num_faces):
            vtx0, vtx1, vtx2 = np.sort(self.faces[i,:])
            
            self.vtx_to_vtx[vtx0].add(vtx1)
            self.vtx_to_vtx[vtx0].add(vtx2)
            self.vtx_to_vtx[vtx1].add(vtx0)
            self.vtx_to_vtx[vtx1].add(vtx2)
            self.vtx_to_vtx[vtx2].add(vtx0)
            self.vtx_to_vtx[vtx2].add(vtx1)

            self.vtx_to_face[vtx0].add(i)
            self.vtx_to_face[vtx1].add(i)
            self.vtx_to_face[vtx2].add(i)

            self.edge_to_face[(vtx0, vtx1)].add(i)
            self.edge_to_face[(vtx0, vtx2)].add(i)
            self.edge_to_face[(vtx1, vtx2)].add(i)
    
if __name__=="__main__":

    fname = "/home/zhenweil/mesh-processing/data/holding_eggs_under_arms.obj"   
    my_mesh = EZMesh(fname)
    
    for i in range(num_faces):
        vtx0, vtx1, vtx2 = faces[i,:]

    new_mesh = vedo.Mesh([vertices, faces])
    vedo.show(new_mesh)
