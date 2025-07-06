import numpy as np
import trimesh
from cache_principal_directions_mod import cache_principal_directions
from cache_boundary_vertices_mod import cache_boundary_vertices

# --- 1. Define 20 random vertices in 3D ---
# Step 1: random 20 points
points = np.random.rand(20, 3)

# Step 2: compute convex hull — returns a watertight mesh
tri_mesh = trimesh.PointCloud(points).convex_hull

_ = cache_boundary_vertices(tri_mesh)

print(_)

# Optional diagnostic prints
print("Mesh has", len(tri_mesh.vertices), "vertices and", len(tri_mesh.faces), "faces")
print("Is watertight?", tri_mesh.is_watertight)

# --- 4. Run your principal-curvature function ---
tensors = cache_principal_directions(tri_mesh)
print("Returned principal directions for", len(tensors), "vertices")
# This list will have 20 entries; boundary vertices (unused) get zero vectors.
print("Sample entry at vertex 0:", tensors[0])

print(tensors)