import numpy as np
import trimesh


def normalize(v, eps=1e-9):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

def simplify_mesh(mesh, target_faces=5000):
    print("Original faces:", len(mesh.faces))
    print("Original vertices:", len(mesh.vertices))

    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)

    # clean up after simplification
    simplified.remove_unreferenced_vertices()
    simplified.fix_normals()

    print("Simplified faces:", len(simplified.faces))
    print("Simplified vertices:", len(simplified.vertices))

    return simplified

def sample_surface_candidates(mesh, n_samples=1000):
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]
    return points, normals, face_ids

def generate_view_candidates(
    mesh,
    n_surface_samples,
    standoff_distances,
    tilt_angles_deg,
):
    points, normals, face_ids = sample_surface_candidates(mesh, n_surface_samples)
    candidates = []

    for p, n, fid in zip(points, normals, face_ids):
        n = normalize(n[None, :])[0]

        for d in standoff_distances:
            for tilt_deg in tilt_angles_deg:
                tilt = np.deg2rad(tilt_deg)

                tangent = np.cross(n, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(tangent) < 1e-6:
                    tangent = np.cross(n, np.array([0.0, 1.0, 0.0]))
                tangent = normalize(tangent[None, :])[0]

                view_dir = normalize(
                    (np.cos(tilt) * (-n) + np.sin(tilt) * tangent)[None, :]
                )[0]

                camera_pos = p - d * view_dir

                candidates.append({
                    "camera_pos": camera_pos,
                    "view_dir": view_dir,       # camera looking direction
                    "target_point": p,
                    "seed_face": fid,
                    "distance": d,
                })

    return candidates

def filter_candidates_by_clearance(mesh_collision, candidates, min_clearance=0.5):
    """
    Reject candidate camera positions that are inside the mesh
    or too close to the mesh surface.

    min_clearance is in mesh units.
    Since your mesh is in cm, min_clearance=0.5 means 0.5 cm.
    """

    if len(candidates) == 0:
        return []

    camera_positions = np.array([c["camera_pos"] for c in candidates])

    # Check if camera points are inside mesh
    if mesh_collision.is_watertight:
        inside = mesh_collision.contains(camera_positions)
    else:
        print("Warning: mesh is not watertight. inside/outside test may be unreliable.")
        inside = np.zeros(len(candidates), dtype=bool)

    # Check nearest distance to mesh surface
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
        mesh_collision,
        camera_positions
    )

    filtered = []

    for c, is_inside, dist in zip(candidates, inside, distances):
        if is_inside:
            continue

        if dist < min_clearance:
            continue

        filtered.append(c)

    print("candidates before clearance filter:", len(candidates))
    print("candidates after clearance filter:", len(filtered))

    return filtered

def visible_faces_from_view(
    camera_pos,
    view_dir,
    ray_intersector,
    face_centers,
    face_normals,
    fov_deg=20,
    max_distance=20.0,
    angle_threshold_deg=70,
    max_rays_per_view=None,
):
    to_faces = face_centers - camera_pos
    distances = np.linalg.norm(to_faces, axis=1)
    dirs = normalize(to_faces)

    # 1. FOV filter
    cos_fov = np.cos(np.deg2rad(fov_deg / 2))
    in_fov = np.dot(dirs, view_dir) > cos_fov

    # 2. Distance filter
    in_range = distances < max_distance

    # 3. Viewing angle filter BEFORE ray casting
    incoming_view = -dirs
    cos_angles = np.sum(incoming_view * face_normals, axis=1)

    cos_angle_threshold = np.cos(np.deg2rad(angle_threshold_deg))
    good_angle = cos_angles > cos_angle_threshold

    valid = in_fov & in_range & good_angle
    valid_face_ids = np.where(valid)[0]

    if len(valid_face_ids) == 0:
        return set()

    # Optional speed cap: useful for debugging
    if max_rays_per_view is not None and len(valid_face_ids) > max_rays_per_view:
        # choose closest faces first
        order = np.argsort(distances[valid_face_ids])
        valid_face_ids = valid_face_ids[order[:max_rays_per_view]]

    ray_origins = np.repeat(camera_pos[None, :], len(valid_face_ids), axis=0)
    ray_dirs = dirs[valid_face_ids]

    hit_faces = ray_intersector.intersects_first(
        ray_origins,
        ray_dirs,
    )

    visible = set()

    for local_idx, hit_face in enumerate(hit_faces):
        if hit_face == -1:
            continue

        expected_face = valid_face_ids[local_idx]

        # If the first hit is the expected face, then it is visible.
        if hit_face == expected_face:
            visible.add(expected_face)

    return visible


def compute_visibility(mesh, candidates):
    valid_candidates = []

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        print("Using Embree ray intersector")
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector
        print("Using triangle ray intersector")

    ray_intersector = RayMeshIntersector(mesh)

    # Precompute these once, not inside every candidate
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    for i, c in enumerate(candidates):
        visible = visible_faces_from_view(
            camera_pos=c["camera_pos"],
            view_dir=c["view_dir"],
            ray_intersector=ray_intersector,
            face_centers=face_centers,
            face_normals=face_normals,
            fov_deg=60,
            max_distance=20.0,
            angle_threshold_deg=70,
            max_rays_per_view=2000,   # set to None for full exact check
        )

        if i % 10 == 0:
            print(
                f"candidate {i}/{len(candidates)}, "
                f"visible faces = {len(visible)}"
            )

        if len(visible) == 0:
            continue

        c["visible_faces"] = visible
        valid_candidates.append(c)

    return valid_candidates

def greedy_select_viewpoints(mesh, candidates, min_new_faces=5):
    uncovered = set(range(len(mesh.faces)))
    selected = []

    while uncovered:
        best = None
        best_gain = 0

        for c in candidates:
            new_faces = c["visible_faces"] & uncovered
            gain = len(new_faces)

            if gain > best_gain:
                best_gain = gain
                best = c

        if best is None or best_gain < min_new_faces:
            break

        selected.append(best)
        uncovered -= best["visible_faces"]

        print(
            f"selected={len(selected)}, "
            f"new_faces={best_gain}, "
            f"uncovered={len(uncovered)}"
        )

    return selected, uncovered

def compute_overall_visibility(mesh, selected):
    total_faces = len(mesh.faces)

    visible_faces = set()

    for c in selected:
        visible_faces |= c["visible_faces"]

    uncovered_faces = set(range(total_faces)) - visible_faces

    # Face-count visibility
    face_visibility = len(visible_faces) / total_faces if total_faces > 0 else 0.0

    # Area-weighted visibility
    face_areas = mesh.area_faces
    total_area = np.sum(face_areas)

    if len(visible_faces) > 0 and total_area > 0:
        visible_ids = np.array(list(visible_faces), dtype=int)
        visible_area = np.sum(face_areas[visible_ids])
        area_visibility = visible_area / total_area
    else:
        area_visibility = 0.0

    return face_visibility, area_visibility, visible_faces, uncovered_faces

def plan_viewpoints(mesh_path):
    mesh_original = trimesh.load(mesh_path, force="mesh")

    mesh_plan = simplify_mesh(mesh_original, target_faces=500)

    candidates = generate_view_candidates(
        mesh_plan,
        n_surface_samples=10,
        standoff_distances=(3, 5, 8),
        tilt_angles_deg=(0, 15, -15, 30, -30),
    )

    print("candidate views:", len(candidates))

    # Important: check candidate camera positions against the ORIGINAL mesh
    candidates = filter_candidates_by_clearance(
        mesh_original,
        candidates,
        min_clearance=0.5,   # cm
    )

    candidates = compute_visibility(mesh_plan, candidates)

    print("visible candidate views:", len(candidates))

    selected, uncovered = greedy_select_viewpoints(
        mesh_plan,
        candidates,
        min_new_faces=5,
    )

    # Add overall visibility info here
    face_visibility, area_visibility, visible_faces, uncovered_faces = compute_overall_visibility(
        mesh_plan,
        selected,
    )

    print("\nOverall visibility on planning mesh:")
    print(f"  Face-count visibility: {face_visibility * 100:.2f}%")
    print(f"  Area-weighted visibility: {area_visibility * 100:.2f}%")
    print(f"  Visible faces: {len(visible_faces)}")
    print(f"  Total faces: {len(mesh_plan.faces)}")
    print(f"  Uncovered faces: {len(uncovered_faces)}")

    viewpoints = np.array([c["camera_pos"] for c in selected])
    view_dirs = np.array([c["view_dir"] for c in selected])

    return viewpoints, view_dirs, selected, uncovered

def make_arrow(start, direction, length=0.05, radius=0.003):
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    # cylinder points along +Z by default
    arrow = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=12,
    )

    # move cylinder so base starts at origin
    arrow.apply_translation([0, 0, length / 2])

    # align +Z to direction
    T = trimesh.geometry.align_vectors([0, 0, 1], direction)
    arrow.apply_transform(T)

    # move to start point
    arrow.apply_translation(start)

    # cone arrow head
    cone = trimesh.creation.cone(
        radius=radius * 3,
        height=length * 0.25,
        sections=12,
    )
    cone.apply_translation([0, 0, length + length * 0.125])
    cone.apply_transform(T)
    cone.apply_translation(start)

    return trimesh.util.concatenate([arrow, cone])

def visualize_views(mesh, viewpoints, view_dirs, arrow_length=0.08):
    scene = trimesh.Scene()

    mesh_vis = mesh.copy()
    mesh_vis.visual.face_colors = [180, 180, 180, 120]
    scene.add_geometry(mesh_vis)

    for p, d in zip(viewpoints, view_dirs):
        sphere = trimesh.creation.uv_sphere(radius=0.2)
        sphere.visual.face_colors = [255, 0, 0, 255]
        sphere.apply_translation(p)
        scene.add_geometry(sphere)

        arrow = make_arrow(
            start=p,
            direction=d,
            length=arrow_length,
            radius=0.05,
        )
        arrow.visual.face_colors = [0, 0, 255, 255]
        scene.add_geometry(arrow)

    scene.show()


if __name__ == "__main__":
    mesh_path = "/home/zhenweil/mesh-processing/data/bunny_holding_eggs_repaired_cm.stl"

    viewpoints, view_dirs, selected, uncovered = plan_viewpoints(mesh_path)

    print("\nNumber of selected views:", len(viewpoints))

    mesh = trimesh.load(mesh_path, force="mesh")
    visualize_views(
        mesh,
        viewpoints,
        view_dirs,
        arrow_length=1,
    )