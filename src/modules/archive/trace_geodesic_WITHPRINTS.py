import numpy as np
import potpourri3d as pp3d
from numpy.linalg import norm

###########################################################################
# Content: 
# The central function in this file is "trace_geodesic".
# 
# Functionality: 
# This function traces a geodesic curve on a triangle mesh starting from a 
# given surface point and a direction vector. It dispatches the tracing 
# algorithm depending on whether the starting point lies on a vertex, edge, 
# or face, and uses potpourri3d's GeodesicTracer to compute the geodesic.
# 
# Application: 
# The function is used to generate geodesic paths across a surface mesh, 
# which are later converted into SurfacePoint instances. These paths can 
# be used in curve evolution and medial axis computation.
#
# Notes:
# - Requires a properly initialized potpourri3d.GeodesicTracer instance.
# - Outputs a list of SurfacePoint objects representing the traced geodesic path.
###########################################################################


# Add the path to your module
import sys
sys.path.append(r"C:\Users\merli\Desktop\BA_thesis\sfc_python_implementation\functions&classes")

# Import your SurfacePoint class
from SurfacePoint_evolved import SurfacePoint


# Helper to extract vertices and faces from trimesh object
def get_vertices_faces(tri_mesh):
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    return vertices, faces


def trace_geodesic(tri_mesh, start_point, direction, tracer, tol=1e-8):
    """
    Dispatch geodesic tracing depending on SurfacePoint type, with debug prints.
    """
    print("\n--- Starting trace_geodesic ---")

    print("Input direction:", direction)
    print("Start point type:", start_point.type)
    
    assert isinstance(direction, np.ndarray), "Direction must be a numpy array"
    assert start_point.type in ['vertex', 'edge', 'face'], f"Unexpected SurfacePoint type: {start_point.type}"
    
    vertices, faces = get_vertices_faces(tri_mesh)

    # Rescale direction to desired length
    length = norm(direction)
    print("Norm of input direction:", length)
    
    if length < tol:
        raise ValueError("Provided direction vector is too small.")
    

    print("Direction norm:", norm(direction))

    # Dispatch depending on SurfacePoint type
    if start_point.type == 'vertex':
        vertex_index = start_point.top_indices[0]
        print("Tracing from vertex index:", vertex_index)
        print("Calling tracer.trace_geodesic_from_vertex()...")
        polyline = tracer.trace_geodesic_from_vertex(vertex_index, direction, max_iterations=1000)
        print("Received polyline from tracer:", polyline)

    else:  # edge or face
        face_index = start_point.face_index
        barycentric_coords = start_point.bary
        print("Tracing from face index:", face_index)
        print("Barycentric coordinates:", barycentric_coords)
        print("Calling tracer.trace_geodesic_from_face()...")
        polyline = tracer.trace_geodesic_from_face(face_index, barycentric_coords, direction, max_iterations=1000)
        print("Received polyline from tracer:", polyline)
    
    # Postprocessing: convert polyline points to SurfacePoint instances
    print("Converting polyline to SurfacePoint instances...")
    surface_points = []
    for i, pt in enumerate(polyline):
        print(f"Converting polyline point {i}: {pt}")
        sp = SurfacePoint.from_position(pt, tri_mesh, tolerance=tol)
        print(f"SurfacePoint {i}: {sp}")
        surface_points.append(sp)

    print("--- Finished trace_geodesic ---\n")
    return surface_points


