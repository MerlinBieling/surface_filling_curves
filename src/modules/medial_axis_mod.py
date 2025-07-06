import numpy as np
from scipy.spatial import KDTree

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

def tracePath(tri_mesh, surfacepoint, d, length, tracer):
    x, y, _ = get_tangent_basis(tri_mesh, surfacepoint)
    traceVec = np.array([np.dot(d, x), np.dot(d, y)])

    # tracer object should include config for maxIters etc., assumed to be passed separately
    res = trace_geodesic(tri_mesh, surfacepoint, traceVec, tracer, tol=1e-8)

    return res


def isIdenticalSurfacePoint(p1, p2):
    return np.allclose(p1.coord3d, p2.coord3d, atol=1e-6)

import numpy as np

def cutTracePath(tri_mesh, path, cartesianPath, _length):
    #path must contain SurfacePoint instances
    # cartesianPath must contain their cartesian coordinates

    length = 0.0
    assert len(path) == len(cartesianPath)

    tracePath = []

    for j in range(1, len(cartesianPath)):
        currentEdgeLen = np.linalg.norm(cartesianPath[j] - cartesianPath[j - 1])
        length += currentEdgeLen

        tracePath.append(path[j - 1])

        if length > _length:
            ratio = (_length - (length - currentEdgeLen)) / currentEdgeLen

            face = sharedFace(path[j - 1], path[j], tri_mesh)

            A, B, C = [tri_mesh.vertices[i] for i in tri_mesh.faces[face]]

            if path[j - 1].face_index == face:
                vec0 = path[j-1].bary
            else:
                vec0 = bary(path[j-1].coord3d, A, B, C)

            # Now the second point...

            if path[j].face_index == face:
                vec1 = path[j].bary
            else:
                vec1 = bary(path[j].coord3d, A, B, C)

            bary = (vec0 * (1 - ratio) + vec1 * ratio)
            nsp = SurfacePoint.from_barycentric(face, bary, tri_mesh, tolerance=1e-6)
            #print(nsp)

            tracePath.append(nsp)
            break

        if j == len(path) - 1:
            tracePath.append(path[-1])

    return tracePath


# THIS DOES NOT WORK YET!!!!
'''
def maximumGeodesicBallRadius(tri_mesh, surfacepoint, nodes, b, i, maxRadius, tracer):

    r = maxRadius
    wholePath = tracePath(tri_mesh, surfacepoint, b, r, tracer)

    cartesianCoords = [sp.coord3d for sp in wholePath]

    

   # p = mmp.traceBack(wholePath[-1])
    finished = isIdenticalSurfacePoint(p[-1], x)

    bsMax, bsMin = 1.0, 0.0
    itrc = 0

    while not finished:
        itrc += 1
        r = maxRadius * (bsMax + bsMin) / 2.0

        path = cutTracePath(tri_mesh, wholePath, cartesianCoords, r)
        #p = mmp.traceBack(path[-1])[-1]

        if isIdenticalSurfacePoint(p, surfacepoint):
            bsMin = (bsMax + bsMin) / 2.0
        else:
            bsMax = (bsMax + bsMin) / 2.0

        if itrc > 10:
            break

    return r
'''
# THIS DOES NOT WORK YET!!!!


# === Euclidean Medial Axis ===
def medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius):
    point_array = np.vstack(cartesianCoords)
    kdtree = KDTree(point_array)

    nodeMedialAxis = [[] for _ in range(len(nodes))]

    for i in range(len(nodes)):
        t = nodeTangents[i]
        n = nodeNormals[i]
        x = np.asarray(cartesianCoords[i], dtype=float)
        #b = np.asarray(nodeBitangents[i], dtype=float)

        b = nodeBitangents[i]


        r_min_plus = min(maximumBallRadius(x, b, i, maxRadius, cartesianCoords, kdtree), maxRadius)
        r_min_minus = min(maximumBallRadius(x, -b, i, maxRadius, cartesianCoords, kdtree), maxRadius)

        medial_minus = x - r_min_minus * b
        medial_plus = x + r_min_plus * b

        nodeMedialAxis[i].append(medial_minus)
        nodeMedialAxis[i].append(medial_plus)

    return nodeMedialAxis # output is list of NumPy arrays of shape (3,)

# === Dispatcher Function ===
def medial_axis(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius, isGeodesic=False):
    if isGeodesic:
        raise NotImplementedError("Geodesic version not implemented in this translation")
    else:
        return medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius)
