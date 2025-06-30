from typing import List, Tuple
import math
import numpy as np

from SurfacePoint_mod import SurfacePoint
from checkAdjacent_mod import checkAdjacent
from sharedFace_mod import sharedFace


EPSILON = 1e-6

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def equals(a: float, b: float) -> bool:
    return abs(a - b) < EPSILON

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

    for i in range(len(segments)):
        v0 = nodes[segments[i][0]]
        v1 = nodes[segments[i][1]]
        path = edgeSurfacePoints[i] # All the surface points on the segment between v0 and v1

        faceIds = []
        pointsOnCurrentFace = []

        for j in range(len(path) + 1):
            v_prev = v0 if j == 0 else path[j - 1]
            v_next = v1 if j == len(path) else path[j]

            # assert(checkAdjacent(v_prev, v_next))
            if not checkAdjacent(v_prev, v_next, tri_mesh):
                continue

            faceId = sharedFace(v_prev, v_next, tri_mesh)

            if len(faceIds) == 0 or faceId != faceIds[-1]:
                faceIds.append(faceId)
                pointsOnCurrentFace.append([v_prev])
            pointsOnCurrentFace[-1].append(v_next)

        assert len(faceIds) == len(pointsOnCurrentFace)

        for j in range(len(faceIds)):
            faceId = faceIds[j]
            if not faceRegistered.get(faceId, False):
                faceRegistered[faceId] = True
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
                        crossProduct = cross(d1, d2)

                        if np.linalg.norm(crossProduct) < EPSILON * aveEdgelen * aveEdgelen:
                            continue

                        diff = q0 - p0
                        denominator = dot(crossProduct, crossProduct)
                        t1 = dot(cross(diff, d2), crossProduct) / denominator
                        t2 = dot(cross(diff, d1), crossProduct) / denominator

                        if (equals(t1, 0) or equals(t1, 1)) and (equals(t2, 0) or equals(t2, 1)):
                            continue

                        if -EPSILON < t1 < 1 + EPSILON and -EPSILON < t2 < 1 + EPSILON:
                            print(f"intersected! t1: {t1}, t2: {t2}, face: {faceId}")
                            return True

    return False
