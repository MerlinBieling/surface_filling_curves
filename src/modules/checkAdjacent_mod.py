import numpy as np

from SurfacePoint_mod import SurfacePoint

def checkAdjacent(pA: SurfacePoint, pB: SurfacePoint, tri_mesh) -> bool:
    """
    Return True if two SurfacePoint inspA.typences are adjacent on the tri_mesh:
    - vertex–vertex: share an edge
    - vertex–edge: vertex part of the edge
    - vertex–face: vertex is one of the face's vertices
    - edge–edge: share a face (i.e., two vertices of one edge are subset of face)
    - edge–face: edge's vertices part of the face's vertices
    - face–face: same face index
    """

    # THIS MIGHT BE A VERY EXPENSIVE FUNCTION (DUE TO TRIMESH TOPOLOGY INFORMATION CALLS)

    def vv(v0, v1):
        return v1.top_indices[0] in tri_mesh.vertex_neighbors[v0.top_indices[0]] # True if vertices who share an edge

    def ve(v, e):
        return v in e.top_indices

    def vf(v, f):
        return v.top_indices[0] in f.top_indices

    def ee(e0, e1):
        # Check if edges share a face

        if e0.face_index == e1.face_index:
            Bool = True
        elif set(e1.top_indices) & set(e0.top_indices):
            x = set(e1.top_indices) ^ set(e0.top_indices)
            if len(x) == 0:
                Bool = True
            elif len(x) == 2:
                it = iter(x)
                a = next(it)
                b = next(it)
                Bool = (b in tri_mesh.vertex_neighbors[a]) # THIS MIGHT BE WRONG, a AND b USED TO BE SWITCHED
            else: 
                Bool = False
        else: 
            Bool = False
        
        return Bool
        

    def ef(e, f): 
        return set(e.top_indices).issubset(set(f.top_indices))
    

    def ff(f0, f1):
        return f0.face_index == f1.face_index

    # Vertex adjacency
    if pA.type == 'vertex' and pB.type == 'vertex':
        return vv(pA, pB)
    if pA.type == 'vertex' and pB.type == 'edge':
        return ve(pA, pB)
    if pA.type == 'edge' and pB.type == 'vertex':
        return ve(pB, pA)

    # Vertex–face
    if pA.type == 'vertex' and pB.type == 'face':
        return vf(pA, pB)
    if pA.type == 'face' and pB.type == 'vertex':
        return vf(pB, pA)

    # Edge–edge
    if pA.type == 'edge' and pB.type == 'edge':
        return ee(pA, pB)

    # Edge–face
    if pA.type == 'edge' and pB.type == 'face':
        return ef(pA, pB)
    if pA.type == 'face' and pB.type == 'edge':
        return ef(pB, pA)

    # Face–face
    if pA.type == 'face' and pB.type == 'face':
        return ff(pA, pB)

    return False
