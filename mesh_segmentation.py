import trimesh
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
from collections import defaultdict

class EZMesh:
    def __init__(self, fname):
        mesh = trimesh.load(fname, process=False)
        mesh.merge_vertices(digits_vertex=5)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        if mesh.is_watertight:
            print("The mesh is closed")
        else:
            print("The mesh has boundaries or holes.")
            print("components:", len(mesh.split(only_watertight=False)))
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

    def segment_based_on_normal(self, normal_thresh):
        segmentation = []
        normals = []
        centroids = []
        face_grouped = [False for _ in range(self.num_faces)]
        order = np.random.permutation(self.num_faces)

        for i in order:
            if face_grouped[i] == True:
                continue
            curr_group = []
            to_be_explored = Queue() 
            to_be_explored.put(i)
            explored = [False for _ in range(self.num_faces)]
            explored[i] = True
            face_grouped[i] = True
            group_normal = self.normals[i]
            normals.append(group_normal)
            curr_face = self.faces[i]
            vtx0 = self.vertices[curr_face[0],:]
            vtx1 = self.vertices[curr_face[1],:]
            vtx2 = self.vertices[curr_face[2],:]
            centroid = (vtx0 + vtx1 + vtx2)/3.0
            centroids.append(centroid)
                        
            while(not to_be_explored.empty()):
                curr_face_idx = to_be_explored.get()
                curr_group.append(curr_face_idx)
                connected_faces = self.get_connected_faces_from_face(curr_face_idx)
                for conn_face_idx in connected_faces:
                    if explored[conn_face_idx] == True or face_grouped[conn_face_idx] == True:
                        continue
                    explored[conn_face_idx] = True
                    next_normal = self.normals[conn_face_idx]
                    angle = angle_between_vec(group_normal, next_normal)
                    if angle < normal_thresh: 
                        to_be_explored.put(conn_face_idx)
                        face_grouped[conn_face_idx] = True

            segmentation.append(curr_group)

        centroids = np.asarray(centroids)
        normals = np.asarray(normals)

        return segmentation, centroids, normals

def angle_between_vec(v1, v2):
    assert (v1.shape == (3,1) and v1.shape==v2.shape) or (v1.shape == (3,) and v1.shape == v2.shape)
    cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_theta))
    return theta_deg

if __name__=="__main__":

    fname = "/home/zhenweil/mesh-processing/data/bunny_holding_eggs_repaired.stl"   
    my_mesh = EZMesh(fname)
    segmentation, centroids, normals = my_mesh.segment_based_on_normal(90)

    print("Number of segmentation: ", len(segmentation))
    
    num_groups = len(segmentation)
    colors = np.random.rand(num_groups, 3)
    face_colors = np.zeros((my_mesh.num_faces, 3), dtype=float)
    for seg_idx, face_indices in enumerate(segmentation):
        face_colors[face_indices] = colors[seg_idx]

    new_mesh = trimesh.Trimesh(vertices=my_mesh.vertices, faces=my_mesh.faces, process=False)
    new_mesh.visual.face_colors = face_colors
    segments = np.stack([centroids, centroids + normals], axis = 1)
    lines = trimesh.load_path(segments)
    scene = trimesh.Scene([new_mesh, lines])
    scene.show()
