import time
from typing import List, Tuple
import numpy as np
import trimesh.triangles as tri
import copy

from get_tangent_basis_mod import get_tangent_basis
from trace_geodesic_mod import trace_geodesic
from sharedFace_mod import sharedFace
from connect_surface_points_mod import connect_surface_points
from modules.archive.remesh_curve_on_surface_mod import remesh_curve_on_surface
from check_intersection_mod import check_intersection
from SurfacePoint_mod import SurfacePoint
from point_point_geodesic_mod import point_point_geodesic

max_iters = 100
shrink = 0.8

def get_new_nodes(
    tri_mesh,
    newNodes,
    tracePathResults,
    tracePathLengths,
    _ratio,
):
    newSurfacePoints = copy.deepcopy(newNodes)
    tracePaths = [[] for _ in newNodes]

    #print(len(newNodes), len(tracePathResults))

    #assert(len(newNodes) == len(tracePathResults))


    for i, res in enumerate(tracePathResults): #res is a list of SurfacePoint instances
        #print(f"\n[get_surface_points] Segment {i}, path length limit: {_ratio * tracePathLengths[i]}")
        if res == None: # This is the case for all them fixed nodes
            # Here we simply leave the same point at newSurfacePoints[i] and tracePaths[i] is an empty list
            continue

        else:
            _length = _ratio * tracePathLengths[i] # this means for every node we go the same fraction of the individual tracePathlength
            length = 0.0

            pathPoints = [sp.coord3d for sp in res]

            #_tracePath = []



            for j in range(1, len(pathPoints)):
                currentEdgeLen = np.linalg.norm(pathPoints[j] - pathPoints[j - 1])
                length += currentEdgeLen
                #print('The updated length is :',length)
                #if type(tracePaths[i]) == None:

                #print(len(res))

                tracePaths[i].append(res[j - 1])
                #_tracePath.append(res[j - 1])

                #print(f"  _ratio {j}, current edge len = {currentEdgeLen:.4f}, accumulated = {length:.4f}")

                if length > _length:
                    ratio = (_length - (length - currentEdgeLen)) / currentEdgeLen
                    face_index = sharedFace(res[j-1], res[j], tri_mesh)
                    assert face_index != -1, "Points do not share a face"

                    #print(face_index)

                    # determining the barycentric coordinates of the two points with regard to the face

                    #A, B, C = [tri_mesh.vertices[tri_mesh.faces[face_index]] for i in tri_mesh.faces[face_index]]¨
                    face_vertices_indices = tri_mesh.faces[face_index]
                    triangle = tri_mesh.vertices[face_vertices_indices]
                    vec0, vec1 = tri.points_to_barycentric(np.array([triangle, triangle]), np.array([res[j-1].coord3d, res[j].coord3d]), method='cross')
                    '''
                    if res[j-1].face_index == face:
                        vec0 = res[j-1].bary
                    else:

                        vec0 = bary(res[j-1].coord3d, A, B, C)
                    # Now the second point...
                    if res[j].face_index == face:
                        vec1 = res[j].bary
                    else:
                        vec1 = bary(res[j].coord3d, A, B, C)
                    '''
                    bary = (vec0 * (1 - ratio) + vec1 * ratio)

                    newSurfacePoints[i] = SurfacePoint.from_barycentric(face_vertices_indices, face_index, bary, tri_mesh, tolerance=1e-6)
                    #print(f"     New SurfPt: {newSurfacePoints[i].coord3d}, type={newSurfacePoints[i].type}")
                    tracePaths[i].append(newSurfacePoints[i])
                    #_tracePath.append(newSurfacePoints[i])
                    break

                if j == len(pathPoints) - 1:
                    newSurfacePoints[i] = res[j] # If the length of the tracePath does not exceed the designated length "_length", then we take the last point of the tracePath as the new Node
                    tracePaths[i].append(newSurfacePoints[i])
                    #_tracePath.append(newSurfacePoints[i])
                    #print(f"  -> Reached final point without exceeding limit.")

                #tracePaths.append(_tracePath)

            #print('Done with this path')

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
    solver,
    dictionary
):

    assert len(direction) == len(nodes)
    # ... additional asserts

    newNodes = copy.deepcopy(nodes)
    newSegments = segments.copy()
    newSegmentSurfacePoints = copy.deepcopy(segmentSurfacePoints)
    newSegmentLengths = segmentLengths.copy()
    newIsFixedNode = isFixedNode.copy()

    tracePathResults = []
    tracePathLengths = []

    for i, node in enumerate(nodes):
        x, y, z = get_tangent_basis(tri_mesh, node)

        d = direction[i] #this is a 3d vector

        if d is not None:
            dx = np.dot(d, x)
            dy = np.dot(d, y)

            # IT MIGHT BE THAT I NEED A SCALING FACTOR OF 0.5 HERE!!!
            #traceVec = 0.5 * (dx * x + dy * y)
            traceVec = (dx * x + dy * y)

            #print('The trace Vector is:',traceVec)
            #print('The trace Vector type is:',traceVec.shape)
    
            #Version without projection of the direction vector into the tangent plane
            #d = direction[i]
            #traceVec = np.array(d).reshape((3, 1))
            _res = trace_geodesic(tri_mesh, node, traceVec, tracer, tol=1e-8)
            #'''
            #-------------------------------------------------------------------
            # THIS BIT IS VERY HEAVY COMPUTATIONALLY; BUT AT LEAST TOPOLOGICALLY CORRECT
            
            last_point = _res[-1]
            res = point_point_geodesic(tri_mesh, meshlib_mesh, node, last_point, solver, dictionary)

            tracePathResults.append(res)
            #-------------------------------------------------------------------
            #'''
            #tracePathResults.append(_res)

            length = np.linalg.norm(traceVec)

            tracePathLengths.append(length)

        else:
            tracePathResults.append(None)
            tracePathLengths.append(None)




    _ratio = 1.0

    for itr in range(max_iters):
        #print('This is iteration number', itr)
        #print(f"\n--- Evolution iteration {itr}, _ratio ratio = {_ratio:.4f} ---")
        #print('before the get_new_nodes the length of isFixedNode and NewNodes are the same?',len(isFixedNode) == len(newNodes))
        newNodes, retractionPath = get_new_nodes(
            tri_mesh, nodes, tracePathResults, tracePathLengths, _ratio
        )
        
        #assert len(isFixedNode) == len(newNodes)
        #print('The lists newNodes and IsFixedNode',newNodes, isFixedNode )
        #t0 = type(newNodes[0])

        #print('This is iteration:', itr)
        #print('t0 has type:',t0)
        #print('Are all the types of newNodes after get_surface_points really surface points?', all(type(x) is t0 for x in newNodes))
        #print('After get_surface_points newNodes is now of size',len(newNodes))
        #print('After get_surface_points tracePathResults is now of size',len(tracePathResults))
        #print('THese are the ouputs of get_surface_points:', newNodes,'The retraction paths', retractionPath)
        
        newSegmentSurfacePoints, newSegmentLengths = connect_surface_points(
            tri_mesh, meshlib_mesh, newNodes, segments, solver, dictionary
        )
        ################################################
        #This is just a check that the required input still has nodes sharing Faces as they should
        for i in range(len(segments)):

            segmentPoints = [newNodes[segments[i][0]]] + newSegmentSurfacePoints[i] + [newNodes[segments[i][1]]]

            for j in range(1, len(segmentPoints)):
                            
                face_index = sharedFace(segmentPoints[j-1], segmentPoints[j], tri_mesh)
                assert face_index != -1, "Points do already not share a face after connect_surface_points"
                #print("Nothing to see here! This works fine!!!")
        ################################################

        print('These are the segments:',segments)
        #print('before the remes_curve_on_surface the length of NewIsFixedNode and NewNodes are the same?',len(isFixedNode) == len(newNodes))
        (newNodes, newSegments,
         newSegmentSurfacePoints, newSegmentLengths,
         newIsFixedNode) = remesh_curve_on_surface(
            tri_mesh, meshlib_mesh, newNodes, segments,
            newSegmentSurfacePoints, newSegmentLengths,
            isFixedNode, h, solver, dictionary
        )
        #print('AFTER the remes_curve_on_surface the length of NewIsFixedNode and NewNodes are the same?',len(newIsFixedNode) == len(newNodes))


        

        #print('After remesh_curve_on_surface newNodes is now of size',len(newNodes))
        #print('After remesh_curve_on_surface tracePathResults is now of size',len(tracePathResults))

        

        intersecting = check_intersection(
            tri_mesh, newNodes, newSegments,
            newSegmentSurfacePoints, newSegmentLengths
        )
        

        if not intersecting:

            
            return (
                newNodes, 
                newSegments,
                newSegmentSurfacePoints,
                newSegmentLengths,
                newIsFixedNode,
                retractionPath
            )

        _ratio *= shrink

    return (
        newNodes, 
        newSegments,
        newSegmentSurfacePoints,
        newSegmentLengths,
        newIsFixedNode,
        retractionPath
    )
