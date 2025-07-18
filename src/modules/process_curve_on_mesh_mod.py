import numpy as np

from SurfacePoint_mod import SurfacePoint

EPSILON = 1e-6

def process_curve_on_mesh(tri_mesh, polylines, _isFixedNode):

    # polyline must look for example like this: polylines = [[0,1,2,3], [4,5,6,4]],
    # where 0,1,2,3,4,5,6 are indices of vertices in tri_mesh.vertices
    # isFixedNode must be a list of booleans corresponding to vertices in tri_mesh.vertices
    # for example: isFixedNode = [0,3,6]

    '''
    # Build V: (n_vertices x 3) numpy array
    #vertexToIndex = here define an empty mapping as a dictionary that takes a vertex as key and returns the index in tri_mesh.vertices
    
    V = np.zeros((len(tri_mesh.vertices), 3), dtype=float)
    for i, vertex in enumerate(tri_mesh.vertices):
        V[i, :] = [vertex[0], vertex[1], vertex[2]]
        # THIS NEXT LITTLE BIT MIGHT POSSIBLY BE WRONG
        #vertexToIndex



    # Build F: (n_faces x 3) integer array
    F = np.zeros((len(tri_mesh.faces), 3), dtype=int)
    for i in range(len(tri_mesh.faces)):
        face_vertices_indices = tri_mesh.faces(i)
        for j in range(3):
            F[i, j] = face_vertices_indices[j]
    '''



    nodes = []
    segments = []
    isFixedNode = []

    isAddedNodeFixed = {}
    addedSegments = []



    # MaYBE THERE NEEDS TO BE ANOTHER LOOP HERE;
    # BECAUSE ISFIXEDNODE MIGHT HAVE TO BE DEFINED LINEWISE IN THE ORIGINAL C++


    for polyline in polylines:
        for i in range(len(polyline)):
                v0 = int(polyline[i - 1])
                v1 = int(polyline[i])
                line = [v0, v1]
                addedSegments.append(line)

    for idx in _isFixedNode:
        isAddedNodeFixed[idx] = True





    # THIS NEXT BIT MIGHT BE WRONG
    nodeAdded = {}
    addedNode2Index = {}
    addedNodeIds = []
    isFixedNode = []

    for i, seg in enumerate(addedSegments):
        if seg[0] == seg[1]:
            addedSegments.pop(i)


    #print(addedSegments)


    for i in range(len(addedSegments)):
        seg = addedSegments[i]
        newSeg = [-1, -1]

        for j in range(2):
            if not nodeAdded.get(seg[j], False):
                vid = len(addedNodeIds)
                addedNodeIds.append(seg[j])

                key = seg[j]
                if key in isAddedNodeFixed:
                    isFixedNode.append(isAddedNodeFixed[key])
                else:
                    isFixedNode.append(False)

                nodeAdded[seg[j]] = True
                addedNode2Index[seg[j]] = vid

            newSeg[j] = addedNode2Index[seg[j]]

        segments.append(newSeg)

    P = np.zeros((len(addedNodeIds), 3))
    for i in range(len(addedNodeIds)):

        vertex_index = addedNodeIds[i]

        faces_indices = tri_mesh.vertex_faces[vertex_index]
        
        faces_indices = [face for face in faces_indices if face != -1]

        face_index = min(faces_indices)

        face_vertices_indices = tri_mesh.faces[face_index]

        x = np.where(face_vertices_indices == vertex_index)[0]

        bary = np.zeros(3)

        bary[x] = 1.0

        sp = SurfacePoint.from_barycentric(face_vertices_indices, face_index, bary, tri_mesh, tolerance=1e-6)

        nodes.append(sp)
    # FROM HERE ON ITS CORRECT

    '''
    for i, seg in enumerate(segments):
         if seg[0] == seg[1]:
              segments.pop(i)

    '''


    #C, _ , I = tri_mesh._pq(P)

    return nodes, segments, isFixedNode



    