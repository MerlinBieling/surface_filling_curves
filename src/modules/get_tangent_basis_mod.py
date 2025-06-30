import numpy as np
import trimesh

###########################################################################
# Content:
# The central function in this file is "get_tangent_basis".
#
# Functionality:
# This function computes an **orthonormal tangent basis** at a given surface 
# point on a triangle mesh. The basis consists of a tangent, bitangent, 
# and normal vector, and adapts to the type of the `SurfacePoint`:
# - At vertices, it uses the vertex normal and computes a tangent in its plane.
# - On edges, it interpolates the normals of the edge vertices and aligns 
#   the tangent with the edge direction.
# - On faces, it uses the face normal and a perpendicular vector in its plane.
#
# Application:
# The function is used in surface_path_evolution to project vectors 
# onto the tangent plane at a point to receive true tangent vectors.
###########################################################################

# Predefine basis vectors to avoid reallocation
BASIS_VECTORS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

def perpendicular_vector(v):
    v = np.asarray(v)
    abs_v = np.abs(v)
    idx = np.argmin(abs_v)  # find component with smallest magnitude
    other = BASIS_VECTORS[idx]
    perp = np.cross(v, other)
    return perp / np.linalg.norm(perp)

def get_tangent_basis(tri_mesh: trimesh.Trimesh, surfacepoint):
    pt_type = surfacepoint.type
    top_indices = surfacepoint.top_indices
    face_index = surfacepoint.face_index

    if pt_type == 'vertex':
        v_idx = top_indices[0]
        normal = tri_mesh.vertex_normals[v_idx]
        tangent = perpendicular_vector(normal)
        bitangent = np.cross(normal, tangent)
        bitangent /= np.linalg.norm(bitangent)
        return tangent, bitangent, normal

    elif pt_type == 'edge':
        v0, v1 = top_indices
        p0 = tri_mesh.vertices[v0]
        p1 = tri_mesh.vertices[v1]
        edge_vec = p1 - p0
        edge_vec /= np.linalg.norm(edge_vec)

        normal1 = tri_mesh.vertex_normals[v0]
        normal2 = tri_mesh.vertex_normals[v1]

        t, index = surfacepoint.t  # interpolation parameter

        if index != v0:
            normal = (1 - t) * normal1 + t * normal2
        else:
            normal = t * normal1 + (1 - t) * normal2
        normal /= np.linalg.norm(normal)

        bitangent = np.cross(normal, edge_vec)
        bitangent /= np.linalg.norm(bitangent)

        return edge_vec, bitangent, normal

    elif pt_type == 'face':
        normal = tri_mesh.face_normals[face_index]
        tangent = perpendicular_vector(normal)
        bitangent = np.cross(normal, tangent)
        bitangent /= np.linalg.norm(bitangent)
        return tangent, bitangent, normal

    else:
        raise ValueError("Invalid surface point type.")