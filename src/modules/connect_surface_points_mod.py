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

# Ensure the local directory is accessible
module_dir = os.path.dirname(os.path.abspath(__file__))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from SurfacePoint_mod import SurfacePoint
from point_point_geodesic_mod import point_point_geodesic

def connect_surface_points(tri_mesh, meshlib_mesh, nodes, segments, solver):
    """
    For each segment (pair of node indices), compute intermediate surface points 
    and geodesic length between them using the geodesic function.

    Parameters:
    - tri_mesh: trimesh.Trimesh object
    - meshlib_mesh: meshlib.mrmeshpy mesh
    - nodes: list of SurfacePoint instances
    - segments: list of [i, j] index pairs into `nodes`

    Returns:
    - edge_surface_points: list of lists of intermediate SurfacePoints (excluding endpoints)
    - edge_lengths: list of geodesic lengths for each segment
    """
    edge_surface_points = []
    edge_lengths = []

    for idx0, idx1 in segments:
        p0 = nodes[idx0]
        p1 = nodes[idx1]

        path, length = point_point_geodesic(tri_mesh, meshlib_mesh, p0, p1, solver)

        if path is None or np.isinf(length):
            #print(f"[WARNING] Geodesic path between nodes {idx0} and {idx1} failed.")
            edge_surface_points.append([])   # fallback: no intermediates
            edge_lengths.append(float('inf'))
            continue

        intermediates = path[1:-1] if len(path) > 2 else []
        edge_surface_points.append(intermediates)
        edge_lengths.append(length)

    return edge_surface_points, edge_lengths
