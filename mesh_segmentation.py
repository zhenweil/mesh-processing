import vedo
import trimesh
import numpy as np
from queue import Queue
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
        self.face_to_face = [set() for _ in range(self.num_faces)]

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
        
        for i in range(self.num_faces):
            vtx0, vtx1, vtx2 = np.sort(self.faces[i,:])
            edges = [(vtx0, vtx1), (vtx0, vtx2), (vtx1, vtx2)]
            for edge in edges:
                conn_faces = self.edge_to_face[edge]
                for conn_face in conn_faces:
                    if conn_face != i:
                        self.face_to_face[i].add(conn_face)
                        
    def get_connected_faces_from_face(self, i):
        return self.face_to_face[i]

    def get_connected_vertices_from_vtx(self, i):
        return self.vtx_to_vtx[i]
    
    def get_connected_faces_from_vtx(self, i):
        return self.vtx_to_face[i]

    def get_connected_faces_from_edge(self, edge):
        ### Edge is a tuple of vertex idx, sorted from small to large
        return self.edge_to_face[edge]

def angle_between_vec(v1, v2):
    assert (v1.shape == (3,1) and v1.shape==v2.shape) or (v1.shape == (3,) and v1.shape == v2.shape)
    cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_theta))
    return theta_deg

if __name__=="__main__":

    fname = "/home/zhenweil/mesh-processing/data/holding_eggs_under_arms.obj"   
    my_mesh = EZMesh(fname)

    face_group = []
    face_grouped = [False for _ in range(my_mesh.num_faces)]
    
    for i in range(my_mesh.num_faces):
        if face_grouped[i] == True:
            continue

        curr_group = []
        to_be_explored = Queue() 
        to_be_explored.put(i)
        explored = [False for _ in range(my_mesh.num_faces)]
        explored[i] = True
        face_grouped[i] = True
        group_normal = my_mesh.normals[i]
                    
        while(not to_be_explored.empty()):
            curr_face_idx = to_be_explored.get()
            curr_group.append(curr_face_idx)
            connected_faces = my_mesh.get_connected_faces_from_face(curr_face_idx)
            for conn_face_idx in connected_faces:
                if explored[conn_face_idx] == True or face_grouped[conn_face_idx] == True:
                    continue
                explored[conn_face_idx] = True
                next_normal = my_mesh.normals[conn_face_idx]
                angle = angle_between_vec(group_normal, next_normal)
                if angle < 20: 
                    to_be_explored.put(conn_face_idx)
                    face_grouped[conn_face_idx] = True

        face_group.append(curr_group)
        print(f"Finished group {len(face_group)}")
        print(f"Current face: {i}, total face: {my_mesh.num_faces}")
    
    #new_mesh = vedo.Mesh([vertices, faces])
    #vedo.show(new_mesh)
