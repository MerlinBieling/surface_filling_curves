import time
from typing import List, Tuple
import numpy as np

from get_tangent_basis_mod import get_tangent_basis
from trace_geodesic_mod import trace_geodesic
from sharedFace_mod import sharedFace
from connect_surface_points_mod import connect_surface_points
from remesh_curve_on_surface_mod import remesh_curve_on_surface
from check_intersection_mod import check_intersection
from SurfacePoint_mod import SurfacePoint

max_iters = 10
shrink = 0.

def bary(P, A, B, C):
    """
    Compute barycentric coordinates (u, v, w) for point P with respect to triangle ABC.
    
    Parameters:
        P, A, B, C: numpy arrays or lists of shape (3,) representing 3D coordinates.

    Returns:
        (u, v, w): tuple of barycentric coordinates such that P = u*A + v*B + w*C and u+v+w = 1
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    P = np.array(P)

    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        raise ValueError("The triangle is degenerate.")

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u, v, w)



def get_surface_points(
    tri_mesh,
    newNodes,
    traceResult,
    traceLengths,
    _ratio,
):
    newSurfacePoints = newNodes.copy()
    tracePaths = [[] for _ in newNodes]

    #print(len(newNodes), len(traceResult))

    #assert(len(newNodes) == len(traceResult))


    for i, res in enumerate(traceResult): #res is a list of SurfacePoint instances
        #print(f"\n[get_surface_points] Segment {i}, path length limit: {_ratio * traceLengths[i]}")
        _length = _ratio * traceLengths[i]
        length = 0.0

        pathPoints = [sp.coord3d for sp in res]

        #_tracePath = []

        if len(res) == 1:

            tracePaths[i].append(newSurfacePoints[i])

            ######################################3333
            ######################################3333
            ######################################3333
            ######################################3333
            ######################################3333
            #DAS IST NOCH NICHT GELéST!!!!!!1
            ######################################3333
            ######################################3333
            ######################################3333
            ######################################3333
            ######################################3333

        else:

            for j in range(1, len(pathPoints)):
                currentEdgeLen = np.linalg.norm(pathPoints[j] - pathPoints[j - 1])
                length += currentEdgeLen
                #if type(tracePaths[i]) == None:

                tracePaths[i].append(res[j - 1])
                #_tracePath.append(res[j - 1])

                #print(f"  _ratio {j}, current edge len = {currentEdgeLen:.4f}, accumulated = {length:.4f}")

                if length > _length:
                    ratio = (_length - (length - currentEdgeLen)) / currentEdgeLen
                    face = sharedFace(res[j-1], res[j], tri_mesh)
                    assert face is not -1, "Points do not share a face"

                    # determining the barycentric coordinates of the two points with regard to the face

                    A, B, C = [tri_mesh.vertices[i] for i in tri_mesh.faces[face]]

                    if res[j-1].face_index == face:
                        vec0 = res[j-1].bary
                    else:
                        vec0 = bary(res[j-1].coord3d, A, B, C)

                    # Now the second point...

                    if res[j].face_index == face:
                        vec1 = res[j].bary
                    else:
                        vec1 = bary(res[j].coord3d, A, B, C)

                    bary = (vec0 * (1 - ratio) + vec1 * ratio)

                    newSurfacePoints[i] = SurfacePoint.from_barycentric(face, bary, tri_mesh, tolerance=1e-6)
                    #print(f"     New SurfPt: {newSurfacePoints[i].coord3d}, type={newSurfacePoints[i].type}")
                    tracePaths[i].append(newSurfacePoints[i])
                    #_tracePath.append(newSurfacePoints[i])
                    break

                if j == len(pathPoints) - 1:
                    newSurfacePoints[i] = res[j]
                    tracePaths[i].append(newSurfacePoints[i])
                    #_tracePath.append(newSurfacePoints[i])
                    #print(f"  -> Reached final point without exceeding limit.")

            #tracePaths.append(_tracePath)

    return newSurfacePoints, tracePaths


def surface_path_evolution(
    tri_mesh,
    meshlib_mesh,
    nodes: List[SurfacePoint],
    segments: List[Tuple[int, int]],
    segmentSurfacePoints: List[List[SurfacePoint]],
    segmentLengths: List[float],
    isFixedNode: List[bool],
    h: float,
    direction,
    tracer,
    solver
):

    assert len(direction) == len(nodes)
    # ... additional asserts

    newNodes = nodes
    traceResult = []
    traceLengths = []

    for i, node in enumerate(nodes):
        x, y, z = get_tangent_basis(tri_mesh, node)

        d = direction[i] #this is a 3d vector

        '''
        # WITH PROJECTION OF THE DIRECTION VECTOR INTO THE TANGENT PLANE AT THE sURFACEPOINT
        dx = np.dot(d, x)
        dy = np.dot(d, y)
        traceVec = np.array([dx, dy])
        '''
  
        #Version without projection of the direction vector into the tangent plane
        d = direction[i]
        traceVec = np.array(d).reshape((3, 1))

        # Now we check that the descent vector is not to small
        if np.linalg.norm(d) < 1e-8:
            print('Very small descent vector, return same point')
            res = [node]
            length = 0
        else:
            print('Large enough descent vector, continue with trace')
            res = trace_geodesic(tri_mesh, node, traceVec, tracer, tol=1e-8)
            length = np.linalg.norm(traceVec)

        traceResult.append(res)
        traceLengths.append(length)

    _ratio = 1.0
    for itr in range(max_iters):
        #print(f"\n--- Evolution iteration {itr}, _ratio ratio = {_ratio:.4f} ---")
        newNodes, retractionPath = get_surface_points(
            tri_mesh, newNodes, traceResult, traceLengths, _ratio
        )
        print('After get_surface_points newNodes is now of size',len(newNodes))
        print('After get_surface_points traceResult is now of size',len(traceResult))

        newSegmentSurfacePoints, newSegmentLengths = connect_surface_points(
            tri_mesh, meshlib_mesh, newNodes, segments, solver
        )

        (newNodes, segments,
         newSegmentSurfacePoints, newSegmentLengths,
         newIsFixedNode) = remesh_curve_on_surface(
            tri_mesh, meshlib_mesh, newNodes, segments,
            newSegmentSurfacePoints, newSegmentLengths,
            isFixedNode, h, solver
        )
        

        #print('After remesh_curve_on_surface newNodes is now of size',len(newNodes))
        #print('After remesh_curve_on_surface traceResult is now of size',len(traceResult))

        

        intersecting = check_intersection(
            tri_mesh, newNodes, segments,
            newSegmentSurfacePoints, newSegmentLengths
        )
        

        if not intersecting:
            
            return (
                newNodes, segments,
                newSegmentSurfacePoints,
                newSegmentLengths,
                newIsFixedNode,
                retractionPath
            )

        _ratio *= shrink

    return (
        newNodes, segments,
        newSegmentSurfacePoints,
        newSegmentLengths,
        newIsFixedNode,
        retractionPath
    )
