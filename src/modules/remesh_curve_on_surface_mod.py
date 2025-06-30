from typing import List, Tuple
import numpy as np
import math

from point_point_geodesic_mod import point_point_geodesic
from SurfacePoint_mod import SurfacePoint
from sharedFace_mod import sharedFace

###########################################################################
# Content:
# The central function in this file is "remesh_curve_on_surface".
#
# Functionality:
# This function refines a piecewise linear curve network on a triangle mesh 
# by applying two core operations:
# (1) **removeShortEdges**: Deletes unnecessary intermediate nodes where 
#     segments are shorter than a given threshold `h`, merging them into 
#     longer segments using updated geodesic paths.
# (2) **subdivideSegments**: Splits segments longer than `2h` into smaller 
#     segments to maintain a uniform edge length distribution.
#
# Application:
# This routine is essential in the `surface_path_evolution` function to 
# preserve **numerical stability and resolution consistency** of curve networks 
# during surface-based optimization and evolution processes.
###########################################################################¨


tol = 1e-8

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

def removeShortEdges(
        tri_mesh, 
        meshlib_mesh,
        nodes, 
        segments, 
        segmentSurfacePoints, 
        segmentLengths, 
        isFixedNode, 
        h,
        solver # The geodesic solver from meshlib solver = mrmesh.GeodesicPathApprox.DijkstraBiDir
        
    ):

    deletingNodes = {i: False for i in range(len(nodes))}

    node2Segments = [[] for _ in range(len(nodes))] # for each node this will hold the segments it is part of

    for i in range(len(segments)):
        node2Segments[segments[i][0]].append(i)
        node2Segments[segments[i][1]].append(i)

    for i in range(len(segments)):
        edgeLen = segmentLengths[i]
        if isFixedNode[segments[i][0]] and isFixedNode[segments[i][1]]:
            continue

        if edgeLen < h:
            shortestLen = float('inf')
            shortestV = -1

            for j in range(2):
                v = segments[i][j]
                if isFixedNode[v]:
                    continue
                    
                if len(node2Segments[v]) != 2:
                    continue

                otherS = node2Segments[v][1] if node2Segments[v][0] == i else node2Segments[v][0]

                if segmentLengths[otherS] < shortestLen and not deletingNodes.get(v, False):
                    shortestLen = segmentLengths[otherS]
                    shortestV = v

            if shortestV != -1:
                deletingNodes[shortestV] = True

    node2NewNode = {}
    newNodes = []
    newNodeIsFixed = []

    for i in range(len(nodes)):
        if deletingNodes[i]:
            continue
        node2NewNode[i] = len(newNodes)
        newNodes.append(nodes[i])
        newNodeIsFixed.append(isFixedNode[i])

    newSegments = []
    newSegmentSurfacePoints = []
    newSegmentLengths = []
    

    for i in range(len(segments)):
        v0, v1 = segments[i]
        if not deletingNodes[v0] and not deletingNodes[v1]:
            newSegments.append([node2NewNode[v0], node2NewNode[v1]])
            newSegmentSurfacePoints.append(segmentSurfacePoints[i])
            newSegmentLengths.append(segmentLengths[i])
        elif not deletingNodes[v0] and deletingNodes[v1]:
            v = v1
            currentSegment = i
            while deletingNodes[v]:
                assert len(node2Segments[v]) == 2
                currentSegment = node2Segments[v][1] if node2Segments[v][0] == currentSegment else node2Segments[v][0]
                v = segments[currentSegment][1] if segments[currentSegment][0] == v else segments[currentSegment][0]

            newSegments.append([node2NewNode[v0], node2NewNode[v]])
            path,_ = point_point_geodesic(tri_mesh, meshlib_mesh, nodes[v0], nodes[v], solver)
            
            length = 0.0
            cartesianCoord = [sp.coord3d for sp in path]
            edgeSurfacePoints = []

            for j in range(len(path)):
                if j != 0:
                    length += np.linalg.norm(cartesianCoord[j] - cartesianCoord[j - 1])
                if j != 0 and j != len(path) - 1:
                    edgeSurfacePoints.append(path[j])

            newSegmentSurfacePoints.append(edgeSurfacePoints)
            newSegmentLengths.append(length)

    assert len(newSegmentSurfacePoints) == len(newSegments)
    assert len(newSegmentLengths) == len(newSegments)

    return newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed

def subdivideSegments( 
    tri_mesh, 
    nodes, 
    segments, 
    segmentSurfacePoints, 
    segmentLengths, 
    isFixedNode, 
    h
    ):

    newNodes = nodes.copy()
    newSegments = segments.copy()
    newSegmentSurfacePoints = segmentSurfacePoints.copy()
    newSegmentLengths = segmentLengths.copy()
    newNodeIsFixed = isFixedNode.copy()

    for i in range(len(segments)):
        edgeLen = segmentLengths[i]
        if isFixedNode[segments[i][0]] and isFixedNode[segments[i][1]]:
            continue

        if edgeLen > 2 * h:
            divisionNum = math.ceil(edgeLen / (2 * h))
            lenPerDivision = edgeLen / divisionNum

            segmentPoints = [nodes[segments[i][0]]] + segmentSurfacePoints[i] + [nodes[segments[i][1]]]
            cartesianCoords = [sp.coord3d for sp in segmentPoints]

            length = 0.0
            newSurfacePoints = []
            newSegmentPoints = [[]]

            for j in range(1, len(cartesianCoords)):
                lenBefore = length
                length += np.linalg.norm(cartesianCoords[j] - cartesianCoords[j - 1])

                numBefore = math.floor(lenBefore / lenPerDivision)
                num = math.floor(length / lenPerDivision)

                for k in range(num - numBefore):
                    _n = numBefore + 1 + k
                    if _n == divisionNum:
                        break

                    ratio = (_n * lenPerDivision - lenBefore) / (length - lenBefore)
                    

                    face = sharedFace(segmentPoints[j - 1], segmentPoints[j], tri_mesh)
                    if face == -1:
                        print(f'sharedFace failed on segment {i} and on point {j} of "segmentPoints"')
                    assert face != -1, f"Points do not share a face"

                    # determining the barycentric coordinates of the two points with regard to the face

                    A, B, C = [tri_mesh.vertices[i] for i in tri_mesh.faces[face]]

                    if segmentPoints[j-1].face_index == face:
                        vec0 = segmentPoints[j-1].bary
                    else:
                        vec0 = bary(segmentPoints[j-1].coord3d, A, B, C)

                    # Now the second point...

                    if segmentPoints[j].face_index == face:
                        vec1 = segmentPoints[j].bary
                    else:
                        vec1 = bary(segmentPoints[j].coord3d, A, B, C)

                    bary = (vec0 * (1 - ratio) + vec1 * ratio)
                    nsp = SurfacePoint.from_barycentric(face, bary, tri_mesh, tolerance=1e-6)
                    #print(nsp)

                    newSurfacePoints.append(nsp)
                    newSegmentPoints.append([])

                if j != len(cartesianCoords) - 1:
                    newSegmentPoints[-1].append(segmentPoints[j])

            assert len(newSegmentPoints) == divisionNum
            assert len(newSurfacePoints) == divisionNum - 1

            newNodeIds = []
            for nsp in newSurfacePoints:
                newNodes.append(nsp)
                newNodeIsFixed.append(False)
                newNodeIds.append(len(newNodes) - 1)

            for j in range(len(newNodeIds) + 1):
                if j == 0:
                    newSegments[i][1] = newNodeIds[j]
                    newSegmentLengths[i] = edgeLen / divisionNum
                    newSegmentSurfacePoints[i] = newSegmentPoints[j]
                elif j == len(newNodeIds):
                    newSegments.append([newNodeIds[j - 1], segments[i][1]])
                    newSegmentLengths.append(edgeLen / divisionNum)
                    newSegmentSurfacePoints.append(newSegmentPoints[j])
                else:
                    newSegments.append([newNodeIds[j - 1], newNodeIds[j]])
                    newSegmentLengths.append(edgeLen / divisionNum)
                    newSegmentSurfacePoints.append(newSegmentPoints[j])

    assert len(newSegmentSurfacePoints) == len(newSegments)
    assert len(newSegmentLengths) == len(newSegments)

    return newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed

def remesh_curve_on_surface(tri_mesh, meshlib_mesh, nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode, h, solver):

    assert len(segmentLengths) == len(segments)
    assert len(segmentSurfacePoints) == len(segments)
    assert len(isFixedNode) == len(nodes)

    newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed = \
        removeShortEdges(tri_mesh, meshlib_mesh, nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode, h, solver)
    
    #print('After removeShortEdges:', len(newNodes))
    newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed = \
        subdivideSegments(tri_mesh, newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed, h)
    #print('After subdivieSegments:', len(newNodes))
    if len(newNodes) < 3 or len(newSegments) < 3:
        return nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode

    return newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed
