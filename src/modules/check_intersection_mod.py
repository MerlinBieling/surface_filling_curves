from typing import List, Tuple
import math
import numpy as np

from SurfacePoint_mod import SurfacePoint
from checkAdjacent_mod import checkAdjacent
from sharedFace_mod import sharedFace


EPSILON = 1e-6





def check_intersection(
    tri_mesh,
    nodes: List[SurfacePoint],
    segments: List[Tuple[int, int]],
    edgeSurfacePoints: List[List[SurfacePoint]],
    edgeLengths: List[float]
):
    
    aveEdgelen = 0.0
    for l in edgeLengths:
        aveEdgelen += l
    aveEdgelen /= len(edgeLengths)

    faceToSurfacePoints = {} 
    faceRegistered = {}       

    # In this first bit we only register which points (both nodes and segmentsurfacepoints)
    # lie on which face. If a face
    for i in range(len(segments)):
        v0 = nodes[segments[i][0]]
        v1 = nodes[segments[i][1]]
        path = edgeSurfacePoints[i] # All the surface points on the segment between v0 and v1

        faceIds = [] # here we gather the face indices of faces which contain surfacepoints which are part of this segment
        pointsOnCurrentFace = []# list of lists, always containing two nodes

        for j in range(len(path) + 1):
            v_prev = v0 if j == 0 else path[j - 1]
            v_next = v1 if j == len(path) else path[j]

            
            #assert(checkAdjacent(v_prev, v_next,  tri_mesh))
            
            if not checkAdjacent(v_prev, v_next, tri_mesh):
                continue
            

            faceId = sharedFace(v_prev, v_next, tri_mesh)

            #assert faceId != -1

            if len(faceIds) == 0 or faceId != faceIds[-1]:
                faceIds.append(faceId)
                pointsOnCurrentFace.append([v_prev])
            pointsOnCurrentFace[-1].append(v_next)

        assert len(faceIds) == len(pointsOnCurrentFace)

        for j in range(len(faceIds)):
            faceId = faceIds[j]
            if not faceRegistered.get(faceId, False): # if the face is not registered yet...
                faceRegistered[faceId] = True # ...register it
                faceToSurfacePoints[faceId] = []
            faceToSurfacePoints[faceId].append(pointsOnCurrentFace[j])

    for faceId, paths in faceToSurfacePoints.items():
        if len(paths) < 2:
            continue
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue

                _paths = [paths[i], paths[j]]
                _cartesianPaths = [[sp.coord3d for sp in _paths[0]], [sp.coord3d for sp in _paths[1]]]

                for l in range(len(_cartesianPaths[0]) - 1):
                    for m in range(len(_cartesianPaths[1]) - 1):
                        p0 = _cartesianPaths[0][l]
                        p1 = _cartesianPaths[0][l + 1]
                        q0 = _cartesianPaths[1][m]
                        q1 = _cartesianPaths[1][m + 1]

                        d1 = p1 - p0
                        d2 = q1 - q0
                        crossProduct = np.cross(d1, d2)

                        if np.linalg.norm(crossProduct) < EPSILON * aveEdgelen * aveEdgelen:
                            continue

                        diff = q0 - p0
                        denominator = np.dot(crossProduct, crossProduct)
                        t1 = np.dot(np.cross(diff, d2), crossProduct) / denominator
                        t2 = np.dot(np.cross(diff, d1), crossProduct) / denominator

                        if (abs(t1 - 0) < EPSILON or abs(t1 - 1) < EPSILON) and (abs(t2 - 0) < EPSILON or abs(t2 - 1) < EPSILON):
                            continue

                        if -EPSILON < t1 < 1 + EPSILON and -EPSILON < t2 < 1 + EPSILON:
                            print(f"intersected! t1: {t1}, t2: {t2}, face: {faceId}")
                            return True

    return False
