import sys
import os
import numpy as np

###########################################################################
# Content:
# The central function in this file is "connect_surface_points".
#
# Functionality:
# Given a graph of node pairs on a surface mesh, it computes the geodesic
# path between each node pair, returns the intermediate SurfacePoint samples,
# and the total geodesic length of the segment. Internally, it uses the
# custom "point_point_geodesic" function.
#
# Application:
# It is used in surface_point_evolution.
###########################################################################

from SurfacePoint_mod import SurfacePoint
from point_point_geodesic_mod import point_point_geodesic

def connect_surface_points(tri_mesh, meshlib_mesh, nodes, segments, solver, dictionary):
    """
    For each segment (pair of node indices), compute intermediate surface points 
    and geodesic length between them using the geodesic function.

    Parameters:
    - tri_mesh: trimesh.Trimesh object
    - meshlib_mesh: meshlib.mrmeshpy mesh
    - nodes: list of SurfacePoint instances
    - segments: list of [i, j] index pairs into `nodes`

    Returns:
    - edgeSurfacePoints: list of lists of intermediate SurfacePoints (excluding endpoints)
    - edgeLengths: list of geodesic lengths for each segment
    """

    numSegments = len(segments)

    # 1. edgeSurfacePoints: Liste von leeren Listen mit Länge numSegments
    edgeSurfacePoints = [[] for _ in range(numSegments)]  # :contentReference[oaicite:1]{index=1}

    # 2. edgeLengths: Liste mit numSegments Einträgen, alle initial auf 0.0
    edgeLengths = [0.0] * numSegments  # :contentReference[oaicite:2]{index=2}

    for i, seg in enumerate(segments):

        idx0, idx1 = seg

        #print(idx0, idx1)
        p0 = nodes[idx0]
        p1 = nodes[idx1]

        path = point_point_geodesic(tri_mesh, meshlib_mesh, p0, p1, solver, dictionary)

        length = 0.0
        cartesianCoord = [sp.coord3d for sp in path]

        #print('p0 and p1 are generated_from:', p0.generated_from, 'and', p1.generated_from)

        for j in range(len(path)):
            if j != 0:
                # Abstand zwischen Kartesischen Koordinaten berechnen und aufsummieren
                diff = cartesianCoord[j] - cartesianCoord[j - 1]
                length += np.linalg.norm(diff)  # entspricht .norm() in C++ :contentReference[oaicite:1]{index=1}

            # Innerhalb (nicht am ersten oder letzten Punkt) den Punkt sammeln
            if j != 0 and j != len(path) - 1:
                edgeSurfacePoints[i].append(path[j])

        edgeLengths[i] = length

    return edgeSurfacePoints, edgeLengths
