

import casadi as ca
import numpy as np
import math
from math import atan2, cos, sin, pi

from get_tangent_basis_mod import get_tangent_basis
from surface_point_vector_field_mod import surface_point_vector_field
from medial_axis_mod import medial_axis
from surface_point_value_mod import surface_point_value


###########################################################################
# Content:
# The central function in this file is "surface_filling_energy_geodesic".
#
# Functionality:
# This function prepares all geometric, topological, and symbolic data 
# necessary to construct and evaluate the surface-filling energy for a 
# network of curves on a mesh. It computes tangent frames, segment 
# classifications, rotation matrices, and symbolic expressions for 
# various energy terms including stretch, field- and curvature-alignedness, 
# bilaplacian smoothness, and medial axis repulsion.
#
# Application:
# "surface_filling_energy_geodesic" is called as a preprocessing and formulation step 
# within the geodesic surface-filling optimization pipeline. It defines the 
# full energy landscape to be minimized and provides the symbolic gradient 
# and Newton descent direction to guide the optimization of node positions 
# on the surface.
###########################################################################

# Dont be spooked by this function. A lot of components come together on this one.

branch_ratio = math.sqrt(math.sqrt(2))

def surface_filling_energy_geodesic(

    # Curve related data:
    nodes,                             # list of SurfacePoint instances REPRESENTING THE CURVE NODES
    cartesianCoords,                  # list of 3D coordinates corresponding to nodes
    segments,                         # list of tuples (or lists) containing node index pairs that form a segment, ??? COULD THIS HAVE THE FORM [[1,2],[2,3]]?
    segmentSurfacePoints,            # list of lists of SurfacePoint instances lying on each segment (excluding endpoints)
    segmentLengths,                  # list of precomputed segment lengths, indices correspond to "segments"
    isFixedNode,                     # boolean list, one per node; True means the node is fixed in optimization

    # Mesh related data:
    tri_mesh,                        # a Trimesh instance used for local tangent basis computation
        
    # Medial Axis related Data:
    maxRadius,                      # maximum search radius for medial axis computation

    # Data Related to the balancing term alpha:
    radius,                        # base strength for medial axis repulsion energy
    useAnisotropicAlphaOnNodes,    # boolean flag: whether to apply node-specific anisotropic alpha scaling
    useAnisotropicAlphaOnMesh,     # boolean flag: whether to interpolate alpha from a mesh texture
    alphaRatioOnNodes,             # list of scaling factors per node, used if "useAnisotropicAlphaOnNodes" is True
    alphaRatioOnMesh,

    # The vector field, as it is retrieved from interpolating the tangent vectors:
    vectorField,

    # Different weights to set the influence of the different energy terms:
    w_fieldAlignedness,            # weight for field-alignedness energy component
    #w_curvatureAlignedness,        # weight for curvature-alignedness energy component
    w_bilaplacian,                 # weight for bilaplacian (second-derivative smoothing) energy component

    p,                             # is used as power to the Shortest curve energy term
    q                              # is used as power in the medial axis energy term
):

    ###########################################################################
    # SETUP
    # Given a segment of the curve that is to be evolved the algorithm 
    # fundamentally differs between such segments where both endpoints are allowed to change 
    # (as is the case for a curve where ALL points may be moved)
    # and such that only one endpoint may change 
    # (as for example is the case for a evoltion of a curve where we have fixed start and end points).
    # The third case, both endpoints being fixed ultimately has no influence on the optimization
    # and thus is ignored (as is the case for example for boundary curves of the underlying mesh).
    # In this first bit of the function we will determine which segments have one or two fixed endpoints 
    # and assign to each segment th according data.
    ###########################################################################

    branch_radius = radius * branch_ratio
    alpha = 4 / (branch_radius ** 2)

    rotationMatrix = [ [np.eye(3), np.eye(3)] for _ in range(len(segments)) ]
    segmentTangent = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]
    segmentNormal = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]
    segmentBitangent = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]

    total_curve_length = sum(segmentLengths)

    for i in range(len(segments)):    
        # Build full list of SurfacePoints on the segment
        pointsOnSegment = [nodes[segments[i][0]]]  # Start node
        for sp in segmentSurfacePoints[i]:
            pointsOnSegment.append(sp)             # Interior segment points
        pointsOnSegment.append(nodes[segments[i][1]])  # End node

        # Convert SurfacePoints to 3D coordinates (as numpy arrays)
        _edgeCartesians = [sp.coord3d for sp in pointsOnSegment]

        # Filter out points that are too close (to remove duplicates)
        # THIS MIGHT NOT FUNCTION SO WELL!!!!!
        edgeCartesians = [_edgeCartesians[0]]
        for j in range(1, len(_edgeCartesians)):
            prev = edgeCartesians[-1]
            if np.linalg.norm(_edgeCartesians[j] - prev) > 1e-6:
                edgeCartesians.append(_edgeCartesians[j])

            #print(_edgeCartesians[j])
            #print('------------------------')
        # Ensure at least two points exist
        if len(edgeCartesians) == 1:
            edgeCartesians.append(_edgeCartesians[-1])

        assert len(edgeCartesians) > 1

        ##################################

        # Compute tangents
        #print(edgeCartesians[1], edgeCartesians[0], edgeCartesians[-1],edgeCartesians[-2])
        tangents = [(edgeCartesians[1] - edgeCartesians[0]),(edgeCartesians[-1] - edgeCartesians[-2])]

        tangents = [t / np.linalg.norm(t) for t in tangents]  # normalize

        for j in range(2):
            v = segments[i][j]
            sp = nodes[v]

            # Get local tangent basis: (x, y, z)
            x,y,z = get_tangent_basis(tri_mesh, sp)   

            # Project tangent onto local tangent plane basis
            
            # THERE MIGHT BE A MISTAKE HERE!
            
            '''
            if sp.type != 'face':
                t_proj = np.dot(x, tangents[j]) * x + np.dot(y, tangents[j]) * y
                tangent_projected = t_proj / np.linalg.norm(t_proj)
            else: 
                tangent_projected = tangents[j]
                tangent_projected = tangent_projected / np.linalg.norm(tangent_projected)
            '''
            #print(tangents[j])
            
            t_proj = np.dot(x, tangents[j]) * x + np.dot(y, tangents[j]) * y
            t_proj *= 1 / np.linalg.norm(t_proj)
            tangent_projected = t_proj
            # Bitangent is orthogonal to both tangent and normal
            #print(z,tangent_projected)
            bitangent_projected = np.cross(z, tangent_projected)

            # Store results
            segmentTangent[i][j] = tangent_projected
            segmentNormal[i][j] = z
            segmentBitangent[i][j] = bitangent_projected

            # Build rotation matrix with columns [tangent, bitangent, normal]
            R = np.column_stack((tangent_projected, bitangent_projected, z))
            rotationMatrix[i][j] = R

    # Initialize storage
    segmentsWith2ActiveNodes = [] # sublist of segments with the according segments
    activeTwoSegment2SegmentIdx = [] # just a list to map indices back to "segments"
    segmentsWith1ActiveNode = [] # sublist of segments with the according segments
    activeOneSegment2SegmentIdx = [] # just a list to map indices back to "segments"
    activeOneSegmentSigns = [] # just a list to keep a sense of direction if only one node is fixed

    # Classify segments
    for i in range(len(segments)):
        v0, v1 = segments[i][0], segments[i][1]

        if isFixedNode[v0] and isFixedNode[v1]: # Both endpoints fixed -> ignore
            continue
        elif not isFixedNode[v0] and not isFixedNode[v1]: # Both endpoints loose
            segmentsWith2ActiveNodes.append((v0, v1))
            activeTwoSegment2SegmentIdx.append(i)
        else: # Only one fixed
            activeNode = v1 if isFixedNode[v0] else v0
            fixedNode = v0 if isFixedNode[v0] else v1
            segmentsWith1ActiveNode.append((activeNode, fixedNode))
            activeOneSegment2SegmentIdx.append(i)
            activeOneSegmentSigns.append(-1 if isFixedNode[v0] else 1)

    # Just two maps to map back and forth between the list of active nodes and "nodes"
    activeNode2NodeIdx = {} 
    node2ActiveNodeIdx = {}

    # populating those lists
    for i in range(len(nodes)):
        if not isFixedNode[i]:
            active_index = len(activeNode2NodeIdx)
            activeNode2NodeIdx[active_index] = i
            node2ActiveNodeIdx[i] = len(node2ActiveNodeIdx)

    # Determining the total curve length:
    total_curve_length = np.sum(segmentLengths)

    # Assertions
    assert len(segmentsWith2ActiveNodes) == len(activeTwoSegment2SegmentIdx)
    assert len(segmentsWith1ActiveNode) == len(activeOneSegment2SegmentIdx)
    assert len(segmentsWith1ActiveNode) == len(activeOneSegmentSigns)

    ###########################################################################
    # ENERGY TERMS
    # Now begins the calculation of the energy function, for the given iteration of the curve. 
    # We do this per-segment. In the end we use Casadi to do the auto-differentiation.
    ###########################################################################

    # List to accumulate energy contributions from all segments
    energy_terms = []

    # Total number of active (movable) nodes
    num_active = len(node2ActiveNodeIdx)

    # Define symbolic variables for each active node
    # x_sym[i] is a 3D CasADi vector representing the optimization variable for the i-th active node
    x_sym = ca.MX.sym('x', 3 * num_active) 

    #x_eval = ca.vertcat(*[ca.DM(cartesianCoords[idx]) for idx in activeNode2NodeIdx])

    ###########################################################################
    # SHORTEST-CURVE TERM and FIELD‑ALIGNED TERM
    # In the following part we determine the first and second term of the energy function: 
    # the field alignment term. 
    # As the vectorfield, to which we want to align the curve,
    # is given per-vertex, we first need to determine what it gives for nodes not on vertices.
    # Then we determine the energy-terms per segment, 
    # differentiating between cases where both segement end points are not fixed 
    # and such where only one is not.
    ###########################################################################

    vectorFieldOnNode = []
    if w_fieldAlignedness > 0:
        for i, node in enumerate(nodes):
            # THESE OPTIONS STILL NEED TO BE REPLACED 
            vf = surface_point_vector_field(tri_mesh, vectorField, node)
            vectorFieldOnNode.append(vf)

    """ 
    Lets leave this out for now
    principalCurvatureOnNode = []
    if w_curvatureAlignedness > 0:
        for i, node in enumerate(nodes):
            # THESE OPTIONS STILL NEED TO BE REPLACED 
            vf, = surface_point_vector_field(tri_mesh, tri_mesh.vertexPrincipalCurvatureDirections, node)
            principalCurvatureOnNode[i] = vf
    """

    # Loop over all segments that have 2 active endpoints
    for e_id in range(len(segmentsWith2ActiveNodes)):

        # This next little bit describes the "unrolling" of a segment into the plane (plain intuition)
        _v0, _v1 = segmentsWith2ActiveNodes[e_id]  
        segmentId = activeTwoSegment2SegmentIdx[e_id] 

        base = 3 * node2ActiveNodeIdx[_v0]
        x0 = x_sym[base     : base + 3] 

        base = 3 * node2ActiveNodeIdx[_v1]
        x1 = x_sym[base     : base + 3]

        #print("x0 type:", type(x0), "value:", x0)
        #print("x1 type:", type(x1), "value:", x1)

        v0 = ca.DM(cartesianCoords[_v0])
        v1 = ca.DM(cartesianCoords[_v1])

        edgeLen = segmentLengths[segmentId]
        R0 = ca.DM(rotationMatrix[segmentId][0])
        R1 = ca.DM(rotationMatrix[segmentId][1])

        # The "unrollled" segment is given by the endpoints p0 and p1
        p0 = ca.mtimes(R0.T, (x0 - v0)) #do x0-v0 to translate the point so the frame's origin is at (0,0,0)
        p1 = ca.mtimes(R1.T, (x1 - v1)) + ca.DM([edgeLen, 0, 0])  
        #print("p0 symbolic:", p0, "has type", type(p0))
        #print("p1 symbolic:", p1, "has type", type(p1))

        # Energy term 1: SHORTEST-CURVE ENERGY
        # Approximate Euclidean distance raised to the power p
        dx = ca.fabs(p0[0] - p1[0])
        dy = ca.fabs(p0[1] - p1[1])
        dz = ca.fabs(p0[2] - p1[2])

        #print("dx symbolic:", dx, "type:", type(dx))
        #print("dy symbolic:", dy, "type:", type(dy))
        #print("dz symbolic:", dz, "type:", type(dz))

        exp = ca.DM(0.5) * ca.DM(p) # the factor 0.5 refers to the root that comes from euclidean norm

        distance_energy = ca.power(dx**2 + dy**2 + dz**2, exp)


        energy = (distance_energy) / (edgeLen * total_curve_length)

        #print(energy.shape)

        # Append to energy term list
        energy_terms.append(energy)


    # Loop over all segments that have 1 active endpoint
    for e_id in range(len(segmentsWith1ActiveNode)):
        _v0, _v1 = segmentsWith1ActiveNode[e_id]   
        segmentId = activeOneSegment2SegmentIdx[e_id]
        sgn = activeOneSegmentSigns[e_id]  # For orientation of the segment

        # Symbolic variable for the active node
        base = 3 * node2ActiveNodeIdx[_v0]
        x0 = x_sym[base     : base + 3] 

        #print("x0 type:", type(x0), "value:", x0)

        # Constant position of the active node
        v0 = ca.DM(cartesianCoords[_v0])

        # Length and rotation matrices (sign-dependent)
        edgeLen = segmentLengths[segmentId]
        R0 = ca.DM(rotationMatrix[segmentId][0 if sgn > 0 else 1])
        R1 = ca.DM(rotationMatrix[segmentId][1 if sgn > 0 else 0])

        # Projected symbolic coordinate and fixed anchor
        p0 = sgn * ca.mtimes(R0.T, (x0 - v0))             # active node (projected)
        p1 = ca.DM([edgeLen, 0, 0])                       # fixed node local coordinate

        #print("p0 symbolic:", p0, "has shape", p0.shape)
        #print("p1 symbolic:", p1, "has shape", p1.shape)

        # Energy term 1: SHORTEST-CURVE ENERGY
        dx = ca.fabs(p0[0] - p1[0])
        dy = ca.fabs(p0[1] - p1[1])
        dz = ca.fabs(p0[2] - p1[2])

        #print("dx symbolic:", dx, "type:", type(dx))
        #print("dy symbolic:", dy, "type:", type(dy))
        #print("dz symbolic:", dz, "type:", type(dz))
        
        distance_energy = ca.power(dx**2 + dy**2 + dz**2, 0.5 * p)

        # Energy term 2: FIELD‑ALIGNED ENERGY
        vf0 = ca.DM(vectorFieldOnNode[_v0])
        vf1 = ca.DM(vectorFieldOnNode[_v1])
        vec0 = ca.mtimes(R0.T, vf0)
        vec1 = ca.mtimes(R1.T, vf1)

        crs0 = w_fieldAlignedness * 0.5 * ca.dot(ca.cross(p1 - p0, vec0), ca.cross(p1 - p0, vec0))
        crs1 = w_fieldAlignedness * 0.5 * ca.dot(ca.cross(p0 - p1, vec1), ca.cross(p0 - p1, vec1))


        energy = (distance_energy) / (edgeLen * total_curve_length)

        #print(energy.shape)

        # Append to the global list of energy terms
        energy_terms.append(energy)

    #print('Shortest Curve and Field Alignment is DONE')


    ###########################################################################
    # MEDIAL-AXIS ENERGY
    # In the following part we determine the fourth term of the energy function: 
    # the medial-axis energy term.
    ###########################################################################

    # Initialize per-node accumulators
    num_nodes = len(nodes)
    nodeTangents = [np.zeros(3) for _ in range(num_nodes)]
    nodeNormals = [np.zeros(3) for _ in range(num_nodes)]
    nodeBitangents = [np.zeros(3) for _ in range(num_nodes)]
    nodeWeight = [0.0 for _ in range(num_nodes)]

    # Loop over all segments and their endpoints
    for i in range(len(segments)):
        for j in range(2):  # j = 0 or 1
            t = segmentTangent[i][j]
            n = segmentNormal[i][j]
            b = segmentBitangent[i][j]
            #print(b)

            v = segments[i][j]  # node index at this endpoint

            # Accumulate normalized frame components
            nodeTangents[v] += t / np.linalg.norm(t)
            nodeNormals[v] += n / np.linalg.norm(n)
            nodeBitangents[v] += b / np.linalg.norm(b)

            # Perturbation for stability in edge cases
            if np.linalg.norm(nodeTangents[v]) < 1e-5:
                nodeTangents[v] -= 0.01 * (n / np.linalg.norm(n))
            if np.linalg.norm(nodeNormals[v]) < 1e-5:
                nodeNormals[v] -= 0.01 * (t / np.linalg.norm(t))
            if np.linalg.norm(nodeBitangents[v]) < 1e-5:
                nodeBitangents[v] -= 0.01 * (t / np.linalg.norm(t))

            # Add half the segment length to the node's weight
            nodeWeight[v] += segmentLengths[i] / 2.0

    # Final normalization step for each node's accumulated vectors
    for i in range(num_nodes):
        nodeTangents[i] = nodeTangents[i] / np.linalg.norm(nodeTangents[i])
        nodeNormals[i] = nodeNormals[i] / np.linalg.norm(nodeNormals[i])
        #print(nodeBitangents[i])
        nodeBitangents[i] = nodeBitangents[i] / np.linalg.norm(nodeBitangents[i])

        #print(type(nodeBitangents[i]))
        #print(type(cartesianCoords[i]))




    # Compute medial axis using provided function
    nodeMedialAxis = medial_axis(nodes, cartesianCoords, 
                                 nodeTangents, nodeNormals, 
                                 nodeBitangents, maxRadius, 
                                 isGeodesic=False)


    # Construct per-node alpha values
    alphas = [0.0] * len(nodes)
    for i in range(len(nodes)):
        if useAnisotropicAlphaOnNodes:  # originally: options.useAnisotropicAlphaOnNodes
            alphas[i] = alphaRatioOnNodes[i] * alpha  # originally: options.alphaRatioOnNodes[i]
        elif useAnisotropicAlphaOnMesh:  # originally: options.useAnisotropicAlphaOnMesh
            alphas[i] = surface_point_value(tri_mesh, alphaRatioOnMesh, nodes[i]) * alpha  # originally: options.alphaRatioOnMesh
        else:
            alphas[i] = alpha

    # ADDING ALSO THIS LAST PART TO THE ENERGY TERMS

    for e_id in range(len(activeNode2NodeIdx)):
        nodeId = activeNode2NodeIdx[e_id]

        # Extract medial-axis centers (shape: (3,))
        c0 = ca.DM(nodeMedialAxis[nodeId][0])
        c1 = ca.DM(nodeMedialAxis[nodeId][1])
        #print(f"[e_id={e_id}] c0 type: {type(c0)}, shape: {c0.shape}, c0: {c0.T}")
        #print(f"[e_id={e_id}] c1 type: {type(c1)}, shape: {c1.shape}, c1: {c1.T}")

        # Extract symbolic variable for this node
        base = 3 * nodeId
        x = x_sym[base     : base + 3]
        #print(f"[e_id={e_id}] x_sym[:,{e_id}] type: {type(x)}, shape: {x.shape}, symbolic: {x}")

        # Symbolic subtraction
        diff0 = x - c0
        diff1 = x - c1
        #print(f"[e_id={e_id}] diff0 type: {type(diff0)}, shape: {diff0.shape}, symbolic: {diff0}")
        #print(f"[e_id={e_id}] diff1 type: {type(diff1)}, shape: {diff1.shape}, symbolic: {diff1}")

        # Norms
        l0 = ca.norm_2(diff0)
        l1 = ca.norm_2(diff1)
        #print(f"[e_id={e_id}] l0 symbolic: {l0}, l1 symbolic: {l1}")

        # Energy term
        d2_sum = l0**2 + l1**2
        repulsion = (alphas[nodeId] * nodeWeight[nodeId] * d2_sum**(q / 2)) / total_curve_length
        #print(f"[e_id={e_id}] repulsion symbolic: {repulsion}")

        #print(repulsion.shape)

        #energy_terms.append(repulsion)

    #print('-------------Medial Axis term is DONE--------------')

    ###########################################################################
    # AND NOW FINALLY TO ADD UP ALL THE ENERGY TERMS
    ###########################################################################

    # Build variable vector x (3D per active node)
    #x = ca.MX.sym("x", 3 * num_active)

    # Build full energy
    #total_energy = sum(energy_terms)

    # After the loop, inspect each term:
    
    # 1. Create the total energy scalar expression:
    #E = ca.sum1(ca.vertcat(*energy_terms))  # or simply `ca.sum(ca.vertcat(...))`

    E = sum(energy_terms)

    #x_flat = x_sym
    #x_flat = ca.vec(x_flat)

    #print(E)

    # 2. Compute Hessian and gradient:

    x_eval = ca.vertcat(*[ca.DM(cartesianCoords[idx]) for idx in activeNode2NodeIdx])

    H, g = ca.hessian(E, x_sym) # this actually computes both hessian and gradient

    #print(type(H), type(g))

    #print(H.depends_on(x), g.depends_on(x_sym))
    #print('This are H and g :',H,g)

    #print("H shape:", H.size1(), H.size2())
    #print("g shape:", g.size1(), g.size2())

    # Solve Newton direction: H * d = -g

    H_func = ca.Function("H_func", [x_sym], [H])
    grad_func = ca.Function("grad_func", [x_sym], [g])

    H_val = H_func(x_eval)
    g_val = grad_func(x_eval)

    #print(H_val.is_constant()) # JUST check if it contains any more symbolic expressions

    H_numpy = np.array(H_val.full())  # full() → NumPy array
    eigvals, eigvecs = np.linalg.eigh(H_numpy)

    # Step 3: Clip eigenvalues to ensure positive definiteness
    delta = 1e-6  # small positive threshold
    eigvals_clipped = np.maximum(eigvals, delta)

    H_posdef = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    d = np.linalg.solve(H_posdef, -np.array(g_val).flatten())
    g = np.array(g_val).flatten()

    # Map results to descent and gradient fields
    descent = [None] * len(cartesianCoords)
    gradient = [None] * len(cartesianCoords)

    # Store per-node vectors
    for i in range(num_active):
        idx = activeNode2NodeIdx[i]
        descent_vec = d[3 * i : 3 * i + 3]
        gradient_vec = g[3 * i : 3 * i + 3]
        descent[idx] = np.array(descent_vec)
        gradient[idx] = np.array(gradient_vec)



    return descent, gradient, ca.Function("E", [x_sym], [E]), nodeMedialAxis

    
