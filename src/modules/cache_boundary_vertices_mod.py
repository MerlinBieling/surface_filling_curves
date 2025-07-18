import numpy as np
import trimesh

from trimesh.grouping import group_rows
from trimesh.proximity import ProximityQuery


def cache_boundary_vertices(tri_mesh):
    """
    Compute a boolean array indicating which vertices lie on the boundary of the mesh.

    Parameters:
    -----------
    tri_mesh : trimesh.Trimesh
        The input triangle mesh.

    Returns:
    --------
    isBoundaryVertex : np.ndarray of bool
        A boolean array where True indicates that the corresponding vertex lies on the boundary.
    """

    # (1) Print number of vertices and faces

    # (2) Get boundary edges (trimesh automatically provides this)
    mask = group_rows(tri_mesh.edges_sorted, require_count=1)
    boundary_edges = tri_mesh.edges[mask]

    # (3) Extract unique vertex indices involved in boundary edges
    boundary_vertices = np.unique(boundary_edges).astype(int)

    # (4) Create a boolean array for all vertices
    isBoundaryVertex = np.zeros(len(tri_mesh.vertices), dtype=bool)
    isBoundaryVertex[boundary_vertices] = True

    # (5) Cache this info into tri_mesh for later usage
    tri_mesh.isBoundaryVertex = isBoundaryVertex

    tri_mesh.isBoundaryEdge = mask

    tri_mesh.boundary_edges = boundary_edges

    tri_mesh._pq = ProximityQuery(tri_mesh)

    return isBoundaryVertex
