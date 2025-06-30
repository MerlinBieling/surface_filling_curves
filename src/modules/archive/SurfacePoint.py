import numpy as np
import trimesh

class SurfacePoint:
    def __init__(self, location_type, top_indices, coord3d, face_indices, face_index, barycentric_coordinates=None):
        """
        Represents a point on a mesh surface.

        location_type: 'vertex', 'edge', or 'face'
        location_index: int or list of ints, depending on type
        coord3d: 3D coordinates of the point
        face_index: index of the face on which this point lies
        barycentric_coordinates: Optional barycentric coordinates (if needed)
        """
        self.type = location_type
        self.top_indices = top_indices
        self.coord3d = coord3d
        self.face_indices = face_indices
        self.face_index = face_index
        self.barycentric = barycentric_coordinates

    @classmethod
    def from_position(cls, point, mesh, tolerance=1e-6):
        """
        Creates a SurfacePoint from a 3D position near/on the mesh.
        
        Parameters:
        - point: a list or array-like of shape (3,) representing a 3D point.
        - mesh: a trimesh.Trimesh object.
        - tolerance: distance threshold for identifying proximity to features.
        """
        point = np.atleast_2d(point)
        if point.shape != (1, 3):
            raise ValueError(f"Point must be a single 3D coordinate with shape (3,) or (1,3), got {point.shape}")
        
        point_single = point[0]  # Extract for computations

        try:
            distance = mesh.nearest.signed_distance(point)[0]
        except Exception:
            distance = np.inf

        # Project onto surface if not already close
        if np.abs(distance) > tolerance:
            locations, distances, face_index = mesh.nearest.on_surface(point)
            location = locations[0]
            face_index = face_index[0]
        else:
            location = point_single
            _, _, face_index = mesh.nearest.on_surface(point)
            face_index = face_index[0]

        face_indices = mesh.faces[face_index]
        #vertex_coords = mesh.vertices[face_indices]


        # Check if near a vertex
        for i, v_idx in enumerate(face_indices):
            if np.linalg.norm(location - mesh.vertices[v_idx]) < tolerance:
                return cls('vertex', [int(v_idx)], location, face_indices, face_index)

        # Check if near an edge
        edges = [(face_indices[i], face_indices[(i+1) % 3]) for i in range(3)]
        for v0, v1 in edges:
            a = mesh.vertices[v0]
            b = mesh.vertices[v1]
            ab = b - a
            t = np.dot(location - a, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            proj = a + t * ab
            if np.linalg.norm(location - proj) < tolerance:
                return cls('edge', [int(v0), int(v1)], location, face_indices, face_index)

        # Otherwise, it's on the face interior
        return cls('face', [int(i) for i in face_indices], location, face_indices, face_index)
