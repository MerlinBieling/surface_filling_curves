import numpy as np
from scipy.spatial import KDTree

# === Closest Point Query ===
def closestPointIndex(p, kdtree):
    """
    Find index of closest point to query point p using KD-tree.
    
    Args:
        p (np.ndarray): query point of shape (3,)
        kdtree (KDTree): pre-built KD-tree

    Returns:
        int: index of closest point
    """
    distance, index = kdtree.query(p, k=1)
    return index


# === Maximum Euclidean Ball Radius ===
def maximumBallRadius(x, b, i, maxRadius, cartesianCoords, kdtree):
    """
    Compute maximal radius of Euclidean ball grown from point x along direction b.

    Args:
        x (np.ndarray): center point of ball (3,)
        b (np.ndarray): direction vector (3,)
        i (int): index of x in point cloud
        maxRadius (float): maximum allowed radius
        cartesianCoords (list of np.ndarray): full dataset of 3D points
        kdtree (KDTree): nearest neighbor KD-tree

    Returns:
        float: maximal radius before touching other points
    """
    r = maxRadius
    c = x + r * b

    nn = closestPointIndex(c, kdtree)
    finished = (nn == i)

    bsMax = 1.0
    bsMin = 0.0
    itrc = 0

    while not finished:
        itrc += 1
        r = maxRadius * (bsMax + bsMin) / 2.0
        c = x + r * b
        nn = closestPointIndex(c, kdtree)

        if nn == i:
            bsMin = (bsMax + bsMin) / 2.0
        else:
            xy = cartesianCoords[nn] - cartesianCoords[i]
            denom = 2 * np.dot(xy, b)
            if denom == 0:
                r = 0.0
            else:
                r = np.linalg.norm(xy)**2 / denom
            c = x + r * b
            nn2 = closestPointIndex(c, kdtree)

            if nn2 == nn or nn2 == i:
                finished = True
            else:
                bsMax = (bsMax + bsMin) / 2.0
                assert bsMax > bsMin

        if itrc > 100:
            break

    return r


# === Euclidean Medial Axis ===
def medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius):
    """
    Compute medial axis using Euclidean distances.

    Args:
        nodes (list): not used in Euclidean case (needed for compatibility)
        cartesianCoords (list of np.ndarray): 3D coordinates of points
        nodeTangents (list of np.ndarray): tangents per point
        nodeNormals (list of np.ndarray): normals per point
        nodeBitangents (list of np.ndarray): bitangents per point
        maxRadius (float): maximal ball radius

    Returns:
        list of list of np.ndarray: medial axis points for each node
    """
    # Build KD-tree
    point_array = np.vstack(cartesianCoords)  # shape (N, 3)
    kdtree = KDTree(point_array)

    nodeMedialAxis = [[] for _ in range(len(nodes))]

    for i in range(len(nodes)):
        t = nodeTangents[i]
        n = nodeNormals[i]
        b = nodeBitangents[i]
        x = cartesianCoords[i]

        r_min_plus = min(maximumBallRadius(x, b, i, maxRadius, cartesianCoords, kdtree), maxRadius)
        r_min_minus = min(maximumBallRadius(x, -b, i, maxRadius, cartesianCoords, kdtree), maxRadius)

        nodeMedialAxis[i].append(x - r_min_minus * b)
        nodeMedialAxis[i].append(x + r_min_plus * b)

    return nodeMedialAxis


# === Dispatcher Function ===
def medial_axis(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius, isGeodesic=False):
    """
    Compute medial axis (Euclidean or Geodesic).

    Args:
        nodes (list): nodes on surface (not used for Euclidean)
        cartesianCoords (list of np.ndarray): 3D coordinates of points
        nodeTangents (list of np.ndarray): tangents per point
        nodeNormals (list of np.ndarray): normals per point
        nodeBitangents (list of np.ndarray): bitangents per point
        maxRadius (float): maximal ball radius
        isGeodesic (bool): flag for geodesic computation

    Returns:
        list of list of np.ndarray: medial axis result
    """
    if isGeodesic:
        raise NotImplementedError("Geodesic version not implemented in this translation")
    else:
        return medial_axis_euclidean(nodes, cartesianCoords, nodeTangents, nodeNormals, nodeBitangents, maxRadius)
