import numpy as np
import math
import trimesh.triangles as tri
import copy

from point_point_geodesic_mod import point_point_geodesic
from SurfacePoint_mod import SurfacePoint
from sharedFace_mod import sharedFace
from find_connected_chains_mod import find_connected_chains

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


tol = 1e-6

def removeShortEdges(
        tri_mesh, 
        meshlib_mesh,
        nodes, 
        segments, 
        segmentSurfacePoints, 
        segmentLengths, 
        isFixedNode, 
        h,
        solver, # The geodesic solver from meshlib solver = mrmesh.GeodesicPathApprox.DijkstraBiDir
        dictionary
        
    ):
    '''
    ################################################
    #This is just a check that the required input still has nodes sharing Faces as they should
    for i in range(len(segments)):

        segmentPoints = [nodes[segments[i][0]]] + segmentSurfacePoints[i] + [nodes[segments[i][1]]]

        for j in range(1, len(segmentPoints)):
                        
            face_index = sharedFace(segmentPoints[j-1], segmentPoints[j], tri_mesh)
            assert face_index != -1, "Points do already not share a face in removeShortEdges"
            #print("Nothing to see here! This works fine!!!")
    ################################################
    '''

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
            path = point_point_geodesic(tri_mesh, meshlib_mesh, nodes[v0], nodes[v], solver, dictionary)
            
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

    newNodes = [sp.copy() for sp in nodes]
    newSegments = copy.deepcopy(segments)
    newSegmentSurfacePoints = [[sp.copy() for sp in sp_list] for sp_list in segmentSurfacePoints]
    newSegmentLengths = copy.deepcopy(segmentLengths)
    newNodeIsFixed = copy.deepcopy(isFixedNode)

    for i in range(len(segments)):
        edgeLen = segmentLengths[i]
        #print('THe curent edgeLen is:',edgeLen)
        if isFixedNode[segments[i][0]] and isFixedNode[segments[i][1]]:
            continue # segments between fixed points and no nodes inbetween are completely ignored

        if edgeLen >  2 * h:
            #print('Still not working?')
            divisionNum = math.ceil(edgeLen / (2 * h))
            lenPerDivision = edgeLen / divisionNum

            segmentPoints = [nodes[segments[i][0]]] + segmentSurfacePoints[i] + [nodes[segments[i][1]]]
            '''
            #print('TEEEST')
            ################################################
            #This is just a check that the required input still has nodes sharing Faces as they should
            for j in range(1, len(segmentPoints)):
                            
                face_index = sharedFace(segmentPoints[j-1], segmentPoints[j], tri_mesh)
                assert face_index != -1, "Points do already not share a face in subdividesegments"
                #print("Nothing to see here! This works fine!!!")
            ################################################
            '''

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
                    

                    face_index = sharedFace(segmentPoints[j-1], segmentPoints[j], tri_mesh)
                    assert face_index != -1, "Points do not share a face"

                    # determining the barycentric coordinates of the two points with regard to the face

                    #A, B, C = [tri_mesh.vertices[tri_mesh.faces[face_index]] for i in tri_mesh.faces[face_index]]¨
                    face_vertices_indices = tri_mesh.faces[face_index]
                    triangle = tri_mesh.vertices[face_vertices_indices]
                    vec0, vec1 = tri.points_to_barycentric(np.array([triangle, triangle]), np.array([segmentPoints[j-1].coord3d, segmentPoints[j].coord3d]), method='cross')
                    #print('Did this bary calculation go well?:')
                    bary = (vec0 * (1 - ratio) + vec1 * ratio)

                    #print('These are the barycentric coordinates', bary)

                    nsp = SurfacePoint.from_barycentric(face_vertices_indices, face_index, bary, tri_mesh, tolerance=1e-6)
                    #print(nsp)

                    newSurfacePoints.append(nsp)
                    newSegmentPoints.append([])

                if j != len(cartesianCoords) - 1:
                    newSegmentPoints[-1].append(segmentPoints[j])

            #print(len(newSegmentPoints), divisionNum)

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
                    newSegments.append([newNodeIds[j-1], segments[i][1]])
                    newSegmentLengths.append(edgeLen / divisionNum)
                    newSegmentSurfacePoints.append(newSegmentPoints[j])
                    
                else:
                    newSegments.append([newNodeIds[j - 1], newNodeIds[j]])
                    newSegmentLengths.append(edgeLen / divisionNum)
                    newSegmentSurfacePoints.append(newSegmentPoints[j])

            #newSegments.append()
                

    assert len(newSegmentSurfacePoints) == len(newSegments)
    assert len(newSegmentLengths) == len(newSegments)

    return newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed





def remesh_curve_on_surface(tri_mesh, meshlib_mesh, nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode, h, solver, dictionary):
    

    newNodes = nodes
    newSegments = segments
    newSegmentSurfacePoints = segmentSurfacePoints
    newSegmentLengths = segmentLengths
    newIsFixedNode = isFixedNode
    '''
    newNodes = [sp.copy() for sp in nodes]
    newSegments = copy.deepcopy(segments)
    newSegmentSurfacePoints = [[sp.copy() for sp in surface_point_list] for surface_point_list in segmentSurfacePoints]
    newSegmentLengths = copy.deepcopy(segmentLengths)
    newIsFixedNode = copy.deepcopy(isFixedNode)
    '''

    assert len(segmentLengths) == len(segments)
    assert len(segmentSurfacePoints) == len(segments)
    assert len(isFixedNode) == len(nodes)

    _newNodes, _newSegments, _newSegmentSurfacePoints, _newSegmentLengths, _newNodeIsFixed = removeShortEdges(tri_mesh, meshlib_mesh, newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newIsFixedNode, h, solver, dictionary)
    
    #print('After removeShortEdges, these are the segments:',_newSegments)
    '''
    print(
    f"_newNodes ({len(_newNodes)} elements): {_newNodes}\n"
    f"_newSegments ({len(_newSegments)} elements): {_newSegments}\n"
    f"_newSegmentSurfacePoints ({len(_newSegmentSurfacePoints)} elements): {_newSegmentSurfacePoints}\n"
    f"_newSegmentLengths ({len(_newSegmentLengths)} elements): {_newSegmentLengths}\n"
    f"_newNodeIsFixed ({len(_newNodeIsFixed)} elements): {_newNodeIsFixed}")
    #
    
    # ---------------FOR DEBUGGIING---------------
    polylines = []
    find_connected_chains(_newSegments, polylines)
    print('Just after removeShortEdges polylines looks now like this:')
    print(len(polylines))
    # ---------------FOR DEBUGGIING---------------
    '''
 

    newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed = subdivideSegments(tri_mesh, _newNodes, _newSegments, _newSegmentSurfacePoints, _newSegmentLengths, _newNodeIsFixed, h)
    '''
    # ---------------FOR DEBUGGIING---------------
    polylines = []
    find_connected_chains(newSegments, polylines)
    print('Just after subdivideSegments newSegments looks now like this:')
    print(len(polylines))
    # ---------------FOR DEBUGGIING---------------
        # ---------------FOR DEBUGGIING---------------
    polylines = []
    find_connected_chains(_newSegments, polylines)
    print('Just after subdivideSegments _newSegments, the input to the function looks like this:')
    print(len(polylines))
    # ---------------FOR DEBUGGIING---------------
    '''

    #print('After subdivieSegments:', newSegments)
    if len(newNodes) < 3 or len(newSegments) < 3:
        return nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode

    return newNodes, newSegments, newSegmentSurfacePoints, newSegmentLengths, newNodeIsFixed
