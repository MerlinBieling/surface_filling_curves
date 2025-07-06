import numpy as np

    
def cache_principal_directions(tri_mesh):
    """Returns the curvature tensors for each vertex of a mesh."""
    tri_mesh.principal_directions = []
    tensors = [] # list of Vector3f
    for i, vertex in enumerate(tri_mesh.vertices):

        # Simply skip over boundary vertices, since they will be part of fixed curves anyway
        if tri_mesh.isBoundaryVertex[i]:
            zero_vec = np.zeros(3)
            tri_mesh.principal_directions.append([zero_vec, zero_vec])
            tensors.append([zero_vec, zero_vec])
            continue

        # get vertex normal as a matrix
        normal = tri_mesh.vertex_normals[i]

        Nvi = np.array([[normal[0]], [normal[1]], [normal[2]]])
        # get sorted 1-ring
        ring = tri_mesh.vertex_neighbors[i]
        # calculate face weightings, wij
        wij = []
        n = len(ring)
        for j in range(n):
            vec0 = tri_mesh.vertices[ring[(j+(n-1))%n]] - vertex
            vec1 = tri_mesh.vertices[ring[j]] - vertex
            vec2 = tri_mesh.vertices[ring[(j+1)%n]] - vertex
            # Assumes closed manifold
            # TODO: handle boundaries
            wij.append(0.5 * (np.linalg.norm(np.cross(vec0, vec1)) + 
                     np.linalg.norm(np.cross(vec1, vec2))))
        wijSum = sum(wij)
        # calculate matrix, Mvi
        Mvi = np.zeros((3, 3))
        I = np.identity(3)
        for j in range(n):
            vec = tri_mesh.vertices[ring[j]] - vertex
            edgeAsMatrix = np.array([[vec[0]], [vec[1]], [vec[2]]]) 
            Tij = (I - Nvi @ Nvi.T) @ edgeAsMatrix
            Tij /= np.linalg.norm(Tij, 'fro')
            kij = (2 * (Nvi.T @ edgeAsMatrix))[0, 0] / np.linalg.norm(vec)**2
            Mvi += (Tij @ Tij.T) * ((wij[j]/wijSum)*kij)
        # get eigenvalues and eigenvectors for Mvi, they are exactly the principal curvatures along with their principal directions
        evals, evecs = np.linalg.eig(Mvi)

        # Suppose evals is a NumPy array of length 3, from np.linalg.eig
        evals = np.real(evals)  # ensure real values due to symmetry

        # Find index of eigenvalue closest to zero
        idx_closest_to_zero = np.argmin(np.abs(evals))

        evals = np.delete(evals, idx_closest_to_zero)
        evecs = np.delete(evecs, idx_closest_to_zero, axis=1)  # axis=1 because columns are eigenvectors

        sort_indices = np.argsort(evals)

        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]

        k1, k2 = evals

        scaling = (k1 - k2)**2
        #scaling = 1
        # replace eigenvector matrix with list of Vector3f
        # Transpose to get eigenvectors as rows (shape becomes (3,)):
        evecs = evecs.T  # Now evecs[i] is the i-th eigenvector as a 1D array of shape (3,)

        #for evecs: reorder the rows so that first row is smaller than second row

        # Normalize each eigenvector
        evecs_normalized = np.array([v / np.linalg.norm(v) for v in evecs])

        # Scale each unit vector by its corresponding eigenvalue
        evecs_scaled = np.array([evecs_normalized[i] * scaling for i in range(2)])
        # sort by absolute value of eigenvalues (norm < min < max)
        # sortv: abs curvature, curvature, Vector3f dir


        tolerance = 1e-6
        # Precompute normalized eigenvectors for clarity
        evecs_norm = [v / (np.linalg.norm(v) + 1e-12) for v in evecs_scaled]

        for j in range(2):
            norm_j = np.linalg.norm(evecs_scaled[j])
            if norm_j < tolerance:
                continue

            dot_sum = 0.0
            count = 0
            relevant_neighbors = [idx for idx in ring if idx < len(tensors)]
            for idx in relevant_neighbors:
                # Ensure neighbor has valid principal direction
                neigh_dir = tensors[idx][0]
                neigh_norm = np.linalg.norm(neigh_dir)
                if neigh_norm < tolerance:
                    continue
                dot = np.dot(neigh_dir / neigh_norm, evecs_norm[j])
                dot_sum += dot
                count += 1

            if count == 0:
                continue  # No neighbors to compare with

            avg_dot = dot_sum / count
            if avg_dot < 0:
                evecs_scaled[j] = -evecs_scaled[j]
                evecs_norm[j] = -evecs_norm[j]



        tensors.append(evecs_scaled)
        tri_mesh.principal_directions.append(evecs_scaled)

    return tensors
    
