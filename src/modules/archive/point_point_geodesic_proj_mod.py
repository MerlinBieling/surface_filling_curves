import sys
import numpy as np
from numpy.linalg import norm
import meshlib.mrmeshpy as mrmesh

###########################################################################
# Content:
# The central function in this file is "point_point_geodesic".
#
# Functionality:
# This function computes the **geodesic path** and its length between two 
# `SurfacePoint` instances on a triangle mesh. It handles three cases:
# 1. If both points lie on the same or subset faces, it returns the Euclidean distance.
# 2. If points lie on adjacent faces, it computes a midpoint based on a weighted 
#    area heuristic across the shared edge.
# 3. For non-adjacent points, it falls back to using the MeshLib geodesic path 
#    approximation via bidirectional Dijkstra.
#
# Application:
# This function is used in the functions "remesh_curve_on_surface", "medial_axis" (in the geodesic case).
###########################################################################


sys.path.append(r"C:\Users\merli\Desktop\BA_thesis\sfc_python_implementation\functions&classes")
from SurfacePoint_mod import SurfacePoint
from sharedFace_mod import sharedFace

'''
def _sharedEdge(p0, p1, tri_mesh):
    if p0.type != 'face' or p1.type != 'face':
        return False  # default null edge

    f0 = p0.face
    f1 = p1.face

    for e0 in f0.adjacentEdges():
        for e1 in f1.adjacentEdges():
            if e0 == e1:
                return e0

    return
'''

def point_point_geodesic(
        tri_mesh, meshlib_mesh, point1, point2, solver, dictionary):

    if sharedFace(point1, point2, tri_mesh) != -1:
        #print("[CASE] Points are subset of each other's face vertices — directly using Euclidean distance")
        euclidean_dist = norm(point1.coord3d - point2.coord3d)
        #print(f"[INFO] Geodesic length: {euclidean_dist}")
        return [point1, point2], euclidean_dist

    adjacency = tri_mesh.face_adjacency
    adjacent_faces = adjacency[(adjacency == point1.face_index).any(axis=1)]
    adjacent_indices = set(adjacent_faces.flatten())
    vertices = [tri_mesh.faces[i] for i in adjacent_indices]

    adj = False
    matching_face = None

    for face in vertices:
        face = [int(v) for v in face]
        face_set = set(face)
        if set(point2.top_indices).issubset(face_set):
            adj = True
            matching_face = face_set
            break

    if adj:
        #print("[CASE] Points lie on adjacent faces")
        intersection = set(point1.face_indices).intersection(matching_face)
        edge = np.sort(list(intersection))

        v0 = np.array(tri_mesh.vertices[edge[0]])
        v1 = np.array(tri_mesh.vertices[edge[1]])
        c0 = np.array(point1.coord3d)
        c1 = np.array(point2.coord3d)

        l0_0 = norm(c0 - v0)
        l0_1 = norm(c0 - v1)
        l1_0 = norm(c1 - v0)
        l1_1 = norm(c1 - v1)
        l = norm(v1 - v0)

        if l0_0 * l == 0 or l0_1 * l == 0 or l1_0 * l == 0 or l1_1 * l == 0:
            return None, np.inf

        def safe_acos(x):
            return np.arccos(np.clip(x, -1.0, 1.0))

        theta0_0 = safe_acos(np.dot(c0 - v0, v1 - v0) / (l0_0 * l))
        theta0_1 = safe_acos(np.dot(c0 - v1, v0 - v1) / (l0_1 * l))
        theta1_0 = safe_acos(np.dot(c1 - v0, v1 - v0) / (l1_0 * l))
        theta1_1 = safe_acos(np.dot(c1 - v1, v0 - v1) / (l1_1 * l))

        theta_0 = theta0_0 + theta1_0
        theta_1 = theta0_1 + theta1_1

        area_tri1 = 0.5 * l0_0 * l1_0 * np.sin(theta_0)
        area_tri2 = 0.5 * l0_1 * l1_1 * np.sin(theta_1)
        total_area = area_tri1 + area_tri2

        if total_area == 0:
            return None, np.inf

        ratio = area_tri1 / total_area

        if not (0 < ratio < 1):
            return None, np.inf

        midpoint_coord = (1 - ratio) * v0 + ratio * v1
        midpoint = SurfacePoint.from_position(midpoint_coord.tolist(), tri_mesh)
        geodesic_length = norm(point1.coord3d - midpoint.coord3d) + norm(midpoint.coord3d - point2.coord3d)
        #print(f"[INFO] Geodesic length: {geodesic_length}")
        return [point1, midpoint, point2], geodesic_length

    else:
       # print("[CASE] Non-adjacent case — using meshlib.")
        x, y, z = point1.coord3d
        mtp1 = mrmesh.findProjection(mrmesh.Vector3f(x, y, z), meshlib_mesh).mtp

        x, y, z = point2.coord3d
        mtp2 = mrmesh.findProjection(mrmesh.Vector3f(x, y, z), meshlib_mesh).mtp

        path_middle = mrmesh.computeGeodesicPath(meshlib_mesh, mtp1, mtp2, solver)
        #for e in path:
            #print('The output type of meshlibs computeGeodesicPath is:',type(e))
        path = [point1]



            

        # The method with projection:

        for i in range(path_middle.__len__()):
            ep = path_middle.__getitem__(i)


            #print('The output type of meshlibs computeGeodesicPath is still:',type(ep))
            
            vec3f = meshlib_mesh.edgePoint(ep)
            sp = SurfacePoint.from_position([vec3f[0], vec3f[1], vec3f[2]], tri_mesh)
            path.append(sp)
        


        #!!!!!!!!!!!!!!!! THIS MIGHT BE VERY WRONG!!!!!!!!!
        path.append(point2)

        geodesic_length = 0.0
        for i in range(len(path) - 1):
            segment_len = norm(np.array(path[i].coord3d) - np.array(path[i + 1].coord3d))
            geodesic_length += segment_len

        #print(f"[INFO] Geodesic length: {geodesic_length}")
        return path, geodesic_length
