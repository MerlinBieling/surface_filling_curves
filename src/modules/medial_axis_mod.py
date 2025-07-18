import numpy as np
from scipy.spatial import KDTree
import trimesh.triangles as tri

from SurfacePoint_mod import SurfacePoint
from get_tangent_basis_mod import get_tangent_basis
from trace_geodesic_mod import trace_geodesic
from sharedFace_mod import sharedFace

###########################################################################
# Content:
# The central functions in this file is "medial_axis".
#
# Functionality:
# This function approximates the **Euclidean medial axis** of a curve network.
# It will return the implicit medial axis point which will
# lie on the surface (in the geodesic sphere case),
# or will be slightly off the surface (in the euclidean case).
#
# Application:
# We need this in the formulation of our Medial-Axis Energy term. 
###########################################################################



# === Maximum Euclidean Ball Radius ===
def maximumBallRadius(x, b, i, maxRadius, cartesianCoords, kdtree):

    x = np.asarray(x, dtype=float)
    #b = np.asarray(b, dtype=float)
    cartesianCoords = np.asarray(cartesianCoords, dtype=float)

    if not isinstance(x, np.ndarray) or x.shape != (3,):
        raise TypeError(f"x must be a numpy array of shape (3,), got {type(x)} with shape {getattr(x, 'shape', None)}")
    if not isinstance(b, np.ndarray) or b.shape != (3,):
        raise TypeError(f"b must be a numpy array of shape (3,), got {type(b)} with shape {getattr(b, 'shape', None)}")
    if not isinstance(maxRadius, (float, int)):
        raise TypeError(f"maxRadius must be a float or int, got {type(maxRadius)}")
    if not isinstance(cartesianCoords, np.ndarray) or cartesianCoords.ndim != 2 or cartesianCoords.shape[1] != 3:
        raise TypeError(f"cartesianCoords must be a numpy array of shape (N, 3), got {type(cartesianCoords)} with shape {getattr(cartesianCoords, 'shape', None)}")
    if not isinstance(kdtree, KDTree):
        raise TypeError(f"kdtree must be an instance of scipy.spatial.cKDTree, got {type(kdtree)}")

    r = maxRadius
    #print(x,r,b)
    c = x + r * b

    # Find nearest neighbor index to c
    nn = kdtree.query(c.reshape(1, 3), k=1)[1][0]
    finished = (nn == i)

    bsMax = 1.0
    bsMin = 0.0
    itrc = 0

    while not finished:
        itrc += 1
        r = maxRadius * (bsMax + bsMin) / 2.0
        c = x + r * b
        nn = kdtree.query(c.reshape(1, 3), k=1)[1][0]

        if nn == i:
            bsMin = (bsMax + bsMin) / 2.0
        else:
            xy = cartesianCoords[nn] - cartesianCoords[i]
            r = float((np.linalg.norm(xy) ** 2) / (2.0 * np.dot(xy, b)))

            c = x + r * b
            nn2 = kdtree.query(c.reshape(1, 3), k=1)[1][0]

            if nn2 == nn or nn2 == i:
                finished = True
            else:
                bsMax = (bsMax + bsMin) / 2.0
                assert bsMax > bsMin

        if itrc > 100:
            break

        #print('r is of the type:',type(r))

    return r

# === Trace Paths ===

def tracePath(tri_mesh, meshlib_mesh, surfacepoint, d, length, tracer, solver, dictionary):
    x, y, _ = get_tangent_basis(tri_mesh, surfacepoint)
    traceVec = np.array([np.dot(d, x), np.dot(d, y)])

    # tracer object should include config for maxIters etc., assumed to be passed separately
    res = trace_geodesic(tri_mesh, meshlib_mesh, surfacepoint, traceVec, tracer, solver, dictionary, tol = 1e-6, use_point_point_geodesic = True)

    return res


def isIdenticalSurfacePoint(p1, p2):
    return np.allclose(p1.coord3d, p2.coord3d, tol=1e-6)

import numpy as np


def cutTracePath(tri_mesh, path, cartesianPath, r): # THIS IS NOT USED YET!!!
    #path must contain SurfacePoint instances
    # cartesianPath must contain their cartesian coordinates

    length = 0.0
    assert len(path) == len(cartesianPath)

    tracePath = []

    for j in range(1, len(cartesianPath)):
        currentEdgeLen = np.linalg.norm(cartesianPath[j] - cartesianPath[j - 1])
        length += currentEdgeLen

        tracePath.append(path[j - 1])

        if length > r:
            ratio = (r - (length - currentEdgeLen)) / currentEdgeLen

            face_index = sharedFace(path[j - 1], path[j], tri_mesh)
            assert face_index != -1, "Points do not share a face"

            # determining the barycentric coordinates of the two points with regard to the face

            #A, B, C = [tri_mesh.vertices[tri_mesh.faces[face_index]] for i in tri_mesh.faces[face_index]]¨
            face_vertices_indices = tri_mesh.faces[face_index]
            triangle = tri_mesh.vertices[face_vertices_indices]
            vec0, vec1 = tri.points_to_barycentric(np.array([triangle, triangle]), np.array([path[j - 1].coord3d, path[j].coord3d]), method='cross')
            #print('Did this bary calculation go well?:')

            bary = (vec0 * (1 - ratio) + vec1 * ratio)

            #print('These are the barycentric coordinates', bary)

            nsp = SurfacePoint.from_barycentric(face_vertices_indices, face_index, bary, tri_mesh, tolerance=1e-6)

            tracePath.append(nsp)
            break

        if j == len(path) - 1:
            tracePath.append(path[-1])

    return tracePath


# THIS DOES NOT WORK YET!!!!
'''
def maximumGeodesicBallRadius(tri_mesh, node, b, i, maxRadius, tracer, distance_map):


    param = 2

    x, y, _ = get_tangent_basis(tri_mesh, node)
    traceVec = np.array([np.dot(b, x), np.dot(b, y)])


    wholePath = tracePath(tri_mesh, meshlib_mesh, surfacepoint, d, length, tracer, solver, dictionary)

    cartesianPath = [sp.coord3d for sp in wholePath]

    cartesianCoords = [sp.coord3d for sp in wholePath]

    last_sp = wholePath[-1]

    vertex_idx = tri_mesh._pq.vertex(last_sp.coord3d)[1][0]

    v0 = tri_mesh.vertices(last_sp.face_indices[0])
    v1 = tri_mesh.vertices(last_sp.face_indices[1])

    edge_len = abs(v0 - v1)

    finished = abs(distance_map[vertex_idx] - np.linalg.norm(traceVec)) < param * edge_len

    
    bsMax = 1.0
    bsMin = 0.0
    itrc = 0

    while not finished:
        itrc += 1
        r = maxRadius * (bsMax + bsMin) / 2.0

        path = cutTracePath(tri_mesh, wholePath, cartesianPath, r)
        c = x + r * b

        vertex_idx = tri_mesh._pq.vertex(path[-1].coord3d)[1][0]

        finished = abs(distance_map[vertex_idx] - np.linalg.norm(traceVec)) < param * edge_len

        if finished:
            bsMin = (bsMax + bsMin) / 2
        else:
            bsMax = (bsMax + bsMin) / 2

        if itrc > 10:
            break


    return r

# THIS DOES NOT WORK YET!!!!



def medial_axis_geodesic(tri_mesh, nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius, tracer):
   
    heat = tri_mesh.heat

    source_points = []
    bool_flags = []

    pp3d_edges_index_map = tri_mesh.pp3d_edges_index_map

    for node in nodes:
        type = node.type
        top_indices = node.top_indices
        face_index = node.face_index
        bary = node.bary

        bool_flags.append(False)

        if type == 'vertex':
            point_data = [(top_indices[0],[])]
        elif type == 'edge':
            _t, v_idx = node.t
            if v_idx == sorted(top_indices)[0]:
                t = _t # THIS IS A TEST AND WILL LIKELY BE WRONG!!!!!!!!!!
            else:
                t = 1 - _t 
            point_data = [(int(pp3d_edges_index_map.get(tuple(sorted(top_indices)))),[float(t)])]
        else:
            point_data = [(int(face_index), [float(x) for x in bary])]

        print(type)
        print(point_data)

        def validate(point_data):
            if not (isinstance(point_data, list) and
                    all(isinstance(p, tuple) and len(p)==2 and
                        isinstance(p[0], int) and
                        isinstance(p[1], list) and
                        all(isinstance(x, float) for x in p[1])
                        for p in point_data)):
                raise TypeError("point_data must be list[tuple[int, list[float]]]")
            return point_data
        
        validate(point_data)

        
        source_points.append(point_data)



    print('These are the source points',source_points)

    distance_map = heat.compute_distance(source_points, [False for _ in source_points])

    nodeMedialAxis = [[] for _ in range(len(nodes))]

    for i in range(len(nodes)):
        t = nodeTangents[i]
        n = nodeNormals[i]
        x = np.asarray(cartesianCoords[i], dtype=float)
        #b = np.asarray(nodeBitangents[i], dtype=float)

        node = nodes[i]

        b = nodeBitangents[i]

        #print(b)

        #print('These are the variables:',x, b, i, maxRadius, cartesianCoords, kdtree)


        r_min_plus = min(maximumGeodesicBallRadius(node, b, i, maxRadius, cartesianCoords, tracer, distance_map), maxRadius)
        r_min_minus = min(maximumGeodesicBallRadius(node, -b, i, maxRadius, cartesianCoords, tracer, distance_map), maxRadius)

        medial_minus = x - r_min_minus * b
        medial_plus = x + r_min_plus * b

        nodeMedialAxis[i].append(medial_minus)
        nodeMedialAxis[i].append(medial_plus)

    return nodeMedialAxis

'''

# === Euclidean Medial Axis ===
def medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius):
    point_array = np.stack(cartesianCoords)
    kdtree = KDTree(point_array)

    nodeMedialAxis = [[] for _ in range(len(nodes))]

    for i in range(len(nodes)):
        t = nodeTangents[i]
        n = nodeNormals[i]
        x = np.asarray(cartesianCoords[i], dtype=float)
        #b = np.asarray(nodeBitangents[i], dtype=float)

        b = nodeBitangents[i]

        #print(b)

        #print('These are the variables:',x, b, i, maxRadius, cartesianCoords, kdtree)


        r_min_plus = min(maximumBallRadius(x, b, i, maxRadius, cartesianCoords, kdtree), maxRadius)
        r_min_minus = min(maximumBallRadius(x, -b, i, maxRadius, cartesianCoords, kdtree), maxRadius)

        medial_minus = x - r_min_minus * b
        medial_plus = x + r_min_plus * b

        nodeMedialAxis[i].append(medial_minus)
        nodeMedialAxis[i].append(medial_plus)

    return nodeMedialAxis # output is list of NumPy arrays of shape (3,)
        
        


# === Dispatcher Function ===
def medial_axis(tri_mesh, nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius, tracer, isGeodesic):
    if isGeodesic:
        return 'This still doesnt work yet'#medial_axis_geodesic(tri_mesh, nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius, tracer)
    else:
        return medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius)
