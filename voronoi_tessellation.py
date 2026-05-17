import heapq
import numpy as np
import trimesh
import matplotlib.pyplot as plt


class ConstrainedMeshVoronoi:
    def __init__(self, mesh, k, normal_weight, seed=None):
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force="mesh")

        self.mesh = mesh
        self.k = k
        self.normal_weight = normal_weight
        self.rng = np.random.default_rng(seed)

        self.centers = self.mesh.triangles_center
        self.normals = self.mesh.face_normals
        self.edges = self.mesh.face_adjacency
        self.n_faces = len(self.mesh.faces)

        self.adj = self.build_adjacency()

        self.seeds = None
        self.labels = None
        self.dist = None

    def build_adjacency(self):
        adj = [[] for _ in range(self.n_faces)]

        for f1, f2 in self.edges:
            spatial = np.linalg.norm(self.centers[f1] - self.centers[f2]) # Euclidean distance between two faces

            normal_penalty = 1.0 - np.clip(np.dot(self.normals[f1], self.normals[f2]), -1.0, 1.0)

            w = spatial * (1.0 + self.normal_weight * normal_penalty)

            adj[f1].append((f2, w))
            adj[f2].append((f1, w))

        return adj

    def constrained_voronoi(self, seeds):
        labels = -np.ones(self.n_faces, dtype=int)
        dist = np.full(self.n_faces, np.inf)

        pq = []

        for region_id, seed in enumerate(seeds):
            labels[seed] = region_id
            dist[seed] = 0.0
            heapq.heappush(pq, (0.0, seed, region_id))

        # Use dijkstra to assign face to the seed
        while pq:
            d, u, region_id = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, w in self.adj[u]:
                nd = d + w

                if nd < dist[v]:
                    dist[v] = nd
                    labels[v] = region_id
                    heapq.heappush(pq, (nd, v, region_id))

        return labels, dist

    def update_seeds_medoid(self, labels):
        new_seeds = []

        for region_id in range(self.k):
            region = np.where(labels == region_id)[0]

            if len(region) == 0:
                new_seeds.append(None)
                continue

            c = self.centers[region].mean(axis=0)

            best = region[
                np.argmin(np.linalg.norm(self.centers[region] - c, axis=1))
            ]

            new_seeds.append(best)

        return np.array(new_seeds, dtype=object)

    def compute_face_curvature(self):
        curvature = np.zeros(self.n_faces)

        for f1, f2 in self.edges:
            dot = np.clip(
                np.dot(self.normals[f1], self.normals[f2]),
                -1.0,
                1.0,
            )

            angle = np.arccos(dot)

            curvature[f1] += angle
            curvature[f2] += angle

        return curvature

    def fit(self, max_iter=30):
        curvature = self.compute_face_curvature()

        prob = curvature + 1e-6
        prob = prob / prob.sum()

        self.seeds = self.rng.choice(
            self.n_faces,
            self.k,
            replace=False,
            p=prob,
        )

        for _ in range(max_iter):
            labels, dist = self.constrained_voronoi(self.seeds)
            new_seeds = self.update_seeds_medoid(labels)

            for i in range(self.k):
                if new_seeds[i] is None:
                    new_seeds[i] = np.argmax(dist)

            new_seeds = new_seeds.astype(int)

            if np.all(new_seeds == self.seeds):
                break

            self.seeds = new_seeds

        self.labels, self.dist = self.constrained_voronoi(self.seeds)

        segmentation = []
        centroids = []
        normals = []

        for region_id in range(self.k):
            region = np.where(self.labels == region_id)[0]

            if len(region) == 0:
                continue

            segmentation.append(region.tolist())

            centroid = self.centers[region].mean(axis=0)
            centroids.append(centroid)

            normal = self.normals[region].mean(axis=0)
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)

        centroids = np.asarray(centroids)
        normals = np.asarray(normals)

        return segmentation, centroids, normals

    def color_mesh(self):
        if self.labels is None:
            raise RuntimeError("Call fit() before color_mesh().")

        colors = plt.cm.tab20(
            self.labels / max(self.labels.max(), 1)
        )[:, :3]

        self.mesh.visual.face_colors = (colors * 255).astype(np.uint8)

        return self.mesh

    def show(self):
        self.color_mesh()
        self.mesh.show()

if __name__=="__main__":
    segmenter = ConstrainedMeshVoronoi(
        "/home/zhenweil/mesh-processing/data/bunny_holding_eggs_repaired.stl",
        k=50,
        normal_weight=50,
        seed=1,
    )

    labels = segmenter.fit(max_iter=30)
    segmenter.show()

    #TODO: how to choose view points based on segmentation? what if there is collision in normal direction?
    # how to properly choose number of regions
    # try sampling based approach