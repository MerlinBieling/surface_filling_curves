import numpy as np
from SurfacePoint_mod import SurfacePoint

def surface_point_to_cartesian(surface_points, vertices=None, faces=None):
    """
    Given a list of SurfacePoint instances, return a list of their 3D coordinates.
    """
    cartesian_coords = [sp.coord3d for sp in surface_points]
    return cartesian_coords
