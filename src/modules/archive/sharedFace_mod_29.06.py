

from SurfacePoint_mod import SurfacePoint


def sharedFace(pA: SurfacePoint, pB: SurfacePoint, tri_mesh) -> int:
    """
    Returns the index of a face that both SurfacePoint instances lie on.
    If none is found, returns -1.
    """

    # Helper: get all face indices the SurfacePoint is on
    def get_faces(point):
        if point.type == 'face':
            return {point.face_index}
        elif point.type == 'edge':
            # Get all faces that include both edge vertices
            v0, v1 = point.top_indices
            faces_v0 = set(tri_mesh.vertex_faces[v0])
            faces_v1 = set(tri_mesh.vertex_faces[v1])
            return faces_v0 & faces_v1
        elif point.type == 'vertex':
            return set(tri_mesh.vertex_faces[point.top_indices[0]])
        else:
            return set()

    facesA = get_faces(pA)
    facesB = get_faces(pB)

    shared = facesA & facesB
    if len(shared) == 0:
        res = -1
    elif len(shared) == 1:
        # the lie both on the boundary of the same face (but not on the same edge)
        # or: one inside face, the other on boundary of the same face
        if pA.type == 'face':
            res = pA.face_index
        elif pB.type == 'face':
            res = pB.face_index
        else:
            # THERE MIGHT BE AN ERROR IN THIS BIT
            triangle = set(pA.top_indices).union(set(pB.top_indices))
            assert len(triangle) == 3, f"Union size is {len(triangle)}, expected 3"
            v0, v1, v2 = triangle
            vf = tri_mesh.vertex_faces

            f0 = set(vf[v0][vf[v0] != -1])
            f1 = set(vf[v1][vf[v1] != -1])
            f2 = set(vf[v2][vf[v2] != -1])
            res = f0 & f1 & f2

    elif len(shared) == 2: # they must lie on the same edge 
        # (with optionally one or both at the vertices)
        res = min(pA.face_index, pB.face_index)

    else:
        # they must be both on the same vertex, thus identical
        res = pA.face_index

    return res

'''Maybe Better:
open different if cases: if sp1.face_index == sp2.face_index return 
that index. Elif: check if sp1.top_indices subset of sp2.top_indices
 or the other way around and return the spx.face_index of the spx which
   has a sp.top_indices which is superset. elif both are vertices: 
   determine else: determine symmetric difference '''
