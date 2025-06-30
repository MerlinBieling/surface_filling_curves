import sys
import numpy as np
from numpy.linalg import norm
import meshlib.mrmeshpy as mrmesh

sys.path.append(r"C:\Users\merli\Desktop\BA_thesis\sfc_python_implementation\functions&classes")
from SurfacePoint_mod import SurfacePoint

def trace_geodesic(tri_mesh, meshlib_mesh, point1, direction, distance):

    x, y, z = point1.coord3d
    mtp1 = mrmesh.findProjection(mrmesh.Vector3f(x, y, z), meshlib_mesh).mtp

    # Here generate a Vector3f from direction list:
    vec3f = mrmesh.Vector3f(direction[0], direction[1], direction[2])

    end = mtp1
    path_middle = mrmesh.trackSection(meshlib_mesh, mtp1, end, vec3f, distance)

    path = [point1]

    for i in range(path_middle.__len__()):
        ep = path_middle.__getitem__(i)
        vec3f = meshlib_mesh.edgePoint(ep)
        sp = SurfacePoint.from_position([vec3f[0], vec3f[1], vec3f[2]], tri_mesh)
        path.append(sp)

    geodesic_length = 0.0
    for i in range(len(path) - 1): 
        segment_len = norm(np.array(path[i].coord3d) - np.array(path[i + 1].coord3d))
        geodesic_length += segment_len

    print(f"[INFO] Geodesic length: {geodesic_length}")
    return path, geodesic_length
