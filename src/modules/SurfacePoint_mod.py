import numpy as np
import trimesh
import trimesh.triangles as tri
import copy

class SurfacePoint:
    def __init__(self, location_type, top_indices, coord3d = None, face_indices = None, face_index = None, bary=None, t=None, tri_mesh = None, generated_from = None):
        """
        Represents a point on a mesh surface.

        Parameters:
        - location_type: 'vertex', 'edge', or 'face'
        - top_indices: list of indices corresponding to the topological feature
        - coord3d: 3D coordinates of the point
        - face_indices: indices of the vertices forming the face
        - face_index: index of the face on which this point lies
        - bary: Optional barycentric coordinates
        """
        self.type = location_type
        self.top_indices = top_indices
        self._coord3d = coord3d
        self.face_indices = face_indices
        self.face_index = face_index
        self.bary = bary
        self.t = t # This is only used in the EDGE case and gives you the relative position on the edge.
        self.tri_mesh = tri_mesh
        self.generated_from = generated_from

    def copy(self):
        # Temporarily remove tri_mesh to avoid deepcopying it
        temp_mesh = self.tri_mesh
        self.tri_mesh = None
        new_copy = copy.deepcopy(self)
        self.tri_mesh = temp_mesh
        new_copy.tri_mesh = temp_mesh
        return new_copy

    @classmethod
    def from_position(cls, point, tri_mesh, tolerance=1e-6):
        """
        Creates a SurfacePoint from a 3D position near/on the mesh.

        Parameters:
        - point: a list or array-like of shape (3,) representing a 3D point.
        - tri_mesh: a trimesh.Trimesh object.
        - tolerance: distance threshold for identifying proximity to features.
        """
        # Check if the input is already a SurfacePoint instance
        if isinstance(point, cls):
            print("Input is already a SurfacePoint instance.")
            return point

        # Ensure point is a NumPy array with shape (1, 3)
        np_point = np.atleast_2d(point)
        if np_point.shape != (1, 3):
            raise ValueError(f"Point must be a single 3D coordinate with shape (3,) or (1,3), got {np_point.shape}")

        # Project the point onto the mesh surface
        projected_point, distance, face_index = tri_mesh._pq.on_surface(np_point)
        projected_point = projected_point[0]
        face_index = face_index[0]

        face_vertices_indices = tri_mesh.faces[face_index]
        face_vertices = tri_mesh.vertices[face_vertices_indices]

        # Prepare triangle for barycentric computation
        triangle = face_vertices.reshape((1, 3, 3))
        point_reshaped = projected_point.reshape((1, 3))


        # Compute barycentric coordinates
        bary = tri.points_to_barycentric(triangle, point_reshaped)[0]

        # Determine the location type based on barycentric coordinates
        close_to_zero = np.isclose(bary, 0.0, atol=tolerance)
        num_zero = np.count_nonzero(close_to_zero)

        t = None

        if num_zero == 0:
            location_type = 'face'
            top_indices = face_vertices_indices.tolist()
        elif num_zero == 1:
            location_type = 'edge'
            zero_index = np.where(close_to_zero)[0][0]
            top_indices = [int(idx) for i, idx in enumerate(face_vertices_indices) if i != zero_index]

            # WE USE FOR t SIMPLY THE FIRST NON-ZERO BARYCENTRIC COORDINATE !!!
            if abs(bary[0]) > tolerance:
                t = bary[0], face_vertices_indices[0]
            elif abs(bary[1]) > tolerance:
                t = bary[1], face_vertices_indices[1]
            else:
                print('Something went very wrong in the determiantion of the t value in surfacePoint')
        elif num_zero == 2:
            location_type = 'vertex'
            vertex_index = face_vertices_indices[np.where(~close_to_zero)[0][0]]
            top_indices = [int(vertex_index)]
        else:
            # All barycentric coordinates are zero; this should not happen
            raise ValueError("Invalid barycentric coordinates: all components are zero.")

        return cls(location_type, top_indices, projected_point, face_vertices_indices, face_index, bary=bary, t=t, tri_mesh=tri_mesh, generated_from = 'position')
    
    @classmethod
    def from_barycentric(cls, face_vertices_indices, face_index, bary, tri_mesh, tolerance=1e-6):
        # IMPORTANT: THE BARYCENTRIC COORDINATES 'bary' MUST REFER TO 'face_vertices_indices'
        # the bary coordinates ALWAYS need to be NORMALIZED!
        """
        Creates a SurfacePoint from a face index and barycentric coordinates.
        """
        face_vertices = tri_mesh.vertices[face_vertices_indices]

        # Reconstruct 3D coordinate from barycentric coordinates
        coord3d = bary[0] * face_vertices[0] + bary[1] * face_vertices[1] + bary[2] * face_vertices[2]

        print(type(coord3d))

        # Determine the location type based on barycentric coordinates
        close_to_zero = np.isclose(bary, 0.0, atol=tolerance)
        num_zero = np.count_nonzero(close_to_zero)
        t = None

        if num_zero == 0:
            location_type = 'face'
            top_indices = face_vertices_indices.tolist()
        elif num_zero == 1:
            location_type = 'edge'
            zero_index = np.where(close_to_zero)[0][0]
            top_indices = [int(idx) for i, idx in enumerate(face_vertices_indices) if i != zero_index]

            # WE USE FOR t SIMPLY THE FIRST NON-ZERO BARYCENTRIC COORDINATE !!!
            if abs(bary[0]) > tolerance:
                t = bary[0], face_vertices_indices[0]
            elif abs(bary[1]) > tolerance:
                t = bary[1], face_vertices_indices[1]
            else:
                print('Something went very wrong in the determiantion of the t value in surfacePoint')
        elif num_zero == 2:
            location_type = 'vertex'
            vertex_index = face_vertices_indices[np.where(~close_to_zero)[0][0]]
            top_indices = [int(vertex_index)]
        else:
            print('Something went very wrong in the determiantion of a surfacePoint')

        return cls(location_type, top_indices, coord3d, face_vertices_indices, face_index, bary, t, tri_mesh=tri_mesh, generated_from = 'bary')
    
    '''
    @classmethod
    def from_top_indices_and_bary(cls, top_indices, bary, tri_mesh, tolerance=1e-6):
        """
        Creates a SurfacePoint from a face index and barycentric coordinates.
        """
        face_vertices = tri_mesh.vertices[top_indices]

        # Reconstruct 3D coordinate from barycentric coordinates
        coord3d = bary[0] * face_vertices[0] + bary[1] * face_vertices[1] + bary[2] * face_vertices[2]

        # Determine the location type based on barycentric coordinates
        close_to_zero = np.isclose(bary, 0.0, atol=tolerance)
        num_zero = np.count_nonzero(close_to_zero)
        t = None

        if len(top_indices) == 3:
            location_type = 'face'

        elif len(top_indices) == 2:
            location_type = 'edge'
            zero_index = np.where(close_to_zero)[0][0]
            if bary[0] != 0:
                t = bary[0], top_indices[0]
            else:
                t = bary[1], top_indices[1]
        elif len(top_indices) == 1:
            location_type = 'vertex'
        else:
            raise ValueError("Invalid barycentric coordinates: all components are zero.")

        return cls(location_type, top_indices, coord3d, bary, t, tri_mesh=tri_mesh)
    '''

    @property
    def coord3d(self):
        if self._coord3d is not None:
            return self._coord3d

        # If not already set, try computing it from bary, face, mesh
        if self.bary is None or self.face_indices is None or self.tri_mesh is None:
            raise ValueError("Cannot compute coord3d: missing bary, face_indices or tri_mesh.")

        face_vertices = self.tri_mesh.vertices[self.face_indices]
        self._coord3d = (
            self.bary[0] * face_vertices[0]
            + self.bary[1] * face_vertices[1]
            + self.bary[2] * face_vertices[2]
        )
        return self._coord3d
    
    @coord3d.setter
    def coord3d(self, value):
        self._coord3d = value
