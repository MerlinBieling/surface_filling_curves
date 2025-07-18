import numpy as np



def perpendicular_vector(v):
    x, y, z = v = np.asarray(v, float)
    s = np.linalg.norm(v)
    sz = z + np.copysign(s, z)
    return np.array([sz*z - x*x, -x*y, -x*sz])




def surface_point_tangent_basis(tri_mesh, sp):

    type = sp.type
    top_indices = sp.top_indices
    face_index = sp.face_index

    if type == 'vertex':
        v_idx = top_indices[0]
        normal = tri_mesh.vertex_normals[v_idx]
        tangent = perpendicular_vector(normal)
        bitangent = np.cross(normal, tangent)
        bitangent /= np.linalg.norm(bitangent)
        return tangent, bitangent, normal

    elif type == 'edge':

        t, v0 = sp.t  # THIS t is the same as sp.bary['index of the according bary']
        v1 = top_indices[0] if v0 != top_indices[0] else top_indices[1]

        p0 = tri_mesh.vertices[v0]
        p1 = tri_mesh.vertices[v1]
        edge_vec = p1 - p0
        edge_vec /= np.linalg.norm(edge_vec)

        normal1 = tri_mesh.vertex_normals[v0]
        normal2 = tri_mesh.vertex_normals[v1]

        normal = (1 - t) * normal2 + t * normal1

        normal /= np.linalg.norm(normal)

        tangent = np.cross(normal, edge_vec)
        tangent /= np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)
        bitangent /= np.linalg.norm(bitangent)

        return tangent, bitangent, normal

    elif type == 'face':

        v0, v1, v2 = sp.face_indices

        normal0 = tri_mesh.vertex_normals[v0]
        normal1 = tri_mesh.vertex_normals[v1]
        normal2 = tri_mesh.vertex_normals[v2]

        b0, b1, b2 = sp.bary

        # Interpolate the normal
        n = b0 * normal0 + b1 * normal1 + b2 * normal2
        n = n / np.linalg.norm(n)  # normalize the result

        _x = tri_mesh.vertices[v0] - sp.coord3d

        x = np.cross(n, _x)
        x /= np.linalg.norm(x)

        y = np.cross(n, x)
        y /= np.linalg.norm(y)
        return x, y, n

    else:
        raise ValueError("Invalid surface point type.")