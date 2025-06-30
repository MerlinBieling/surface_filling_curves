

from SurfacePoint_mod import SurfacePoint


import numpy as np


# This method will NOT work with meshes that are highly volatile and have complex curvature

def sharedFace(pA: SurfacePoint, pB: SurfacePoint, tri_mesh) -> int:
    """
    Returns the index of a face that both SurfacePoint instances lie on.
    If none is found, returns -1.
    """

    # THIS MIGHT BE A VERY EXPENSIVE FUNCTION (DUE TO TRIMESH TOPOLOGY INFORMATION CALLS)

    def vv(v0, v1):

        if v1.top_indices[0] in tri_mesh.vertex_neighbors[v0.top_indices[0]]:

            if v0.top_indices[0] in v1.face_indices:
                res = v1.face_index
            elif v1.top_indices[0] in v0.face_indices:
                res = v0.face_index
            else:# we need a more expensive lookup
                
                faces_adj_v0 = tri_mesh.vertex_faces[v0.top_indices[0]]
                faces_adj_v1 = tri_mesh.vertex_faces[v1.top_indices[0]]

                face_indices = set(faces_adj_v0) & set(faces_adj_v1)

                res = min(i for i in face_indices if i != -1)

        else:
            res = -1 
        
        return res
    
    def ve(v, e):
        v_idx = v.top_indices[0]
        edge_vertices = set(e.top_indices)
        triangle = edge_vertices | {v_idx}

        # Case 1: The vertex lies directly on the edge
        if v_idx in edge_vertices:
            return e.face_index

        # Case 2: Look for a triangle containing both the vertex and the edge
        for i in tri_mesh.vertex_faces[v_idx]:
            if i == -1:
                continue
            face_vertices = set(tri_mesh.faces[i])
            if triangle == face_vertices:
                return i

        return -1

    def vf(v, f):

        if v.top_indices[0] in f.face_indices:
            res = f.face_index
        else:
            res = -1
        return res

    def ee(e0, e1):
        # Check if two edges share a face
        edge0 = set(e0.top_indices)
        edge1 = set(e1.top_indices)

        # If edges are exactly the same
        if edge0 == edge1:
            return e0.face_index

        # Form triangle from both edges
        triangle = edge0 | edge1

        # Use one vertex from e0 to limit face search
        v_idx = e0.top_indices[0]

        for i in tri_mesh.vertex_faces[v_idx]:
            if i == -1:
                continue
            face_vertices = set(tri_mesh.faces[i])
            if triangle == face_vertices:
                return i

        return -1


    def ef(e, f):
        if set(e.top_indices).issubset(set(f.top_indices)):
            res = f.face_index
        else:
            res = -1
        return res
    

    def ff(f0, f1):
        if f0.face_index == f1.face_index:
            res = f0.face_index
        else:
            res = -1
        return res
    
    # Vertex adjacency
    if pA.type == 'vertex' and pB.type == 'vertex':
        return vv(pA, pB)
    elif pA.type == 'vertex' and pB.type == 'edge':
        return ve(pA, pB)
    elif pA.type == 'edge' and pB.type == 'vertex':
        return ve(pB, pA)

    # Vertex–face
    elif pA.type == 'vertex' and pB.type == 'face':
        return vf(pA, pB)
    elif pA.type == 'face' and pB.type == 'vertex':
        return vf(pB, pA)

    # Edge–edge
    elif pA.type == 'edge' and pB.type == 'edge':
        return ee(pA, pB)

    # Edge–face
    elif pA.type == 'edge' and pB.type == 'face':
        return ef(pA, pB)
    elif pA.type == 'face' and pB.type == 'edge':
        return ef(pB, pA)

    # Face–face
    elif pA.type == 'face' and pB.type == 'face':
        return ff(pA, pB)
    
    else:
        return -1

'''Maybe Better:
open different if cases: if sp1.face_index == sp2.face_index return 
that index. Elif: check if sp1.top_indices subset of sp2.top_indices
 or the other way around and return the spx.face_index of the spx which
   has a sp.top_indices which is superset. elif both are vertices: 
   determine else: determine symmetric difference '''
