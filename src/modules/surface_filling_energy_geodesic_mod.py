

import casadi as ca
import numpy as np
import math
from math import atan2, cos, sin, pi
import numpy as np
from scipy.linalg import cholesky, LinAlgError
import copy

from get_tangent_basis_mod import get_tangent_basis
from surface_point_vector_field_mod import surface_point_vector_field
from medial_axis_mod import medial_axis
from surface_point_value_mod import surface_point_value
from surface_point_tangent_basis_mod import surface_point_tangent_basis


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

tol = 1e-6

def populate_node2ActiveNodeIdx_and_activeNode2NodeIdx(nodes, isFixedNode):
    activeNode2NodeIdx = {} 
    node2ActiveNodeIdx = {}
    # populating those lists
    for i in range(len(nodes)):
        if not isFixedNode[i]:
            active_index = len(activeNode2NodeIdx)
            activeNode2NodeIdx[active_index] = i
            node2ActiveNodeIdx[i] = len(node2ActiveNodeIdx)
    return activeNode2NodeIdx, node2ActiveNodeIdx

def get_tan_bitan_normal_and_rotation_matrix(segments, nodes, segmentSurfacePoints, tri_mesh):
    rotationMatrix = [ [np.eye(3), np.eye(3)] for _ in range(len(segments)) ]
    segmentTangent = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]
    segmentNormal = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]
    segmentBitangent = [ [np.zeros(3), np.zeros(3)] for _ in range(len(segments)) ]

    for i in range(len(segments)):    
    # Build full list of SurfacePoints on the segment
        pointsOnSegment = [nodes[segments[i][0]]]  # Start node
        for sp in segmentSurfacePoints[i]:
            pointsOnSegment.append(sp)             # Interior segment points
        pointsOnSegment.append(nodes[segments[i][1]])  # End node

        # Convert SurfacePoints to 3D coordinates (as numpy arrays)
        _edgeCartesians = [sp.coord3d for sp in pointsOnSegment]

        # Filter out points that are too close (to remove duplicates)
        edgeCartesians = [_edgeCartesians[0]]
        for j in range(1, len(_edgeCartesians)):
            prev = edgeCartesians[-1]
            dist = np.linalg.norm(_edgeCartesians[j] - prev)
            #print('These points have distance', dist)
            if dist > tol:
                edgeCartesians.append(_edgeCartesians[j])
        # Ensure at least two points exist
        if len(edgeCartesians) == 1:
            edgeCartesians.append(_edgeCartesians[-1])

        assert len(edgeCartesians) > 1

        ##################################

        # Compute tangents
        #print(edgeCartesians[1], edgeCartesians[0], edgeCartesians[-1],edgeCartesians[-2])
        tangents = [edgeCartesians[1] - edgeCartesians[0],edgeCartesians[-1] - edgeCartesians[-2]]

        tangents = [t / np.linalg.norm(t) for t in tangents]  # normalize

        for j in range(2):
            v = segments[i][j]
            sp = nodes[v]

            # Get local tangent basis: (x, y, z)
            x,y,z = surface_point_tangent_basis(tri_mesh, sp)   

            # Project tangent onto local tangent plane basis
            
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

            #print('This is inside the surfac filling ---:',segmentBitangent[i][j])

            # Build rotation matrix with columns [tangent, bitangent, normal]
            R = np.column_stack((tangent_projected, bitangent_projected, z))
            rotationMatrix[i][j] = R
    return segmentTangent, segmentNormal, segmentBitangent, rotationMatrix

def populate_segments_lists(segments, isFixedNode):
    
    # Initialize storage
    segmentsWith2ActiveNodes = [] # sublist of segments with the according segments
    activeTwoSegment2SegmentIdx = [] # just a list to map indices back to "segments"
    segmentsWith1ActiveNode = [] # sublist of segments with the according segments
    activeOneSegment2SegmentIdx = [] # just a list to map indices back to "segments"
    activeOneSegmentSigns = [] # just a list to keep a sense of direction if only one node is fixed

    # Classify segments
    for i in range(len(segments)):
        v0, v1 = segments[i]

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

    return segmentsWith2ActiveNodes, activeTwoSegment2SegmentIdx, segmentsWith1ActiveNode, activeOneSegment2SegmentIdx, activeOneSegmentSigns

def populate_vector_fields(w_fieldAlignedness, w_curvatureAlignedness, nodes, tri_mesh, vectorField):
    vectorFieldOnNode = []
    if w_fieldAlignedness > 0:
        for i, node in enumerate(nodes):
            vf = surface_point_vector_field(tri_mesh, vectorField, node)
            vectorFieldOnNode.append(vf)

    curvature_field = np.array([
    tri_mesh.principal_directions[i][0]
    for i in range(len(tri_mesh.vertices))])

    principalCurvatureOnNode = []
    if w_curvatureAlignedness > 0:
        for i, node in enumerate(nodes):
            # THESE OPTIONS STILL NEED TO BE REPLACED 
            vf = surface_point_vector_field(tri_mesh, curvature_field, node)
            principalCurvatureOnNode.append(vf) 
    
    #print('BEGINNING',principalCurvatureOnNode, 'ENDING')

    return vectorFieldOnNode, principalCurvatureOnNode


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
    tracer,
        
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
    w_curvatureAlignedness,        # weight for curvature-alignedness energy component
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

    # Getting the length of all the curves to later be able to 
    # give weight to segements according to their relative length
    totalCurveLength = sum(segmentLengths)


    assert len(nodes) == len(isFixedNode), "Mismatch: nodes and isFixedNode must be the same length"
    assert len(nodes) == len(cartesianCoords), "Mismatch: nodes and cartesianCoords must be the same length"

    # Just two maps to map back and forth between the list of active nodes and "nodes"
    activeNode2NodeIdx, node2ActiveNodeIdx = populate_node2ActiveNodeIdx_and_activeNode2NodeIdx(nodes, isFixedNode)

    # Determine segment-wise: tangents, bitangents, normals and rotation matrices
    segmentTangent, segmentNormal, segmentBitangent, rotationMatrix = get_tan_bitan_normal_and_rotation_matrix(segments, nodes, segmentSurfacePoints, tri_mesh)

    # Classify segments
    segmentsWith2ActiveNodes, activeTwoSegment2SegmentIdx, segmentsWith1ActiveNode, activeOneSegment2SegmentIdx, activeOneSegmentSigns = populate_segments_lists(segments, isFixedNode)

    vectorFieldOnNode, principalCurvatureOnNode = populate_vector_fields(w_fieldAlignedness, w_curvatureAlignedness, nodes, tri_mesh, vectorField)

    # More Assertions
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
    

    # Loop over all segments that have 2 active endpoints
    for e_id in range(len(segmentsWith2ActiveNodes)):

        # This next little bit describes the "unrolling" of a segment into the plane (plain intuition)
        _v0, _v1 = segmentsWith2ActiveNodes[e_id]  
        segmentId = activeTwoSegment2SegmentIdx[e_id] 

        base = 3 * node2ActiveNodeIdx[_v0]

        x0 = x_sym[base     : base + 3] 

        base = 3 * node2ActiveNodeIdx[_v1]
        #print('this is the base',base)
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

        # Energy term 2: FIELD‑ALIGNED ENERGY
        #if w_fieldAlignedness > 0:
        vf0 = ca.DM(vectorFieldOnNode[_v0])  if w_fieldAlignedness > tol else ca.DM.zeros(3, 1) # Vector field at node _v0
        vf1 = ca.DM(vectorFieldOnNode[_v1])  if w_fieldAlignedness > tol else ca.DM.zeros(3, 1)  # Vector field at node _v1
        vec0 = ca.mtimes(R0.T, vf0)          # Transform vector into local frame
        vec1 = ca.mtimes(R1.T, vf1)

        crs0 = w_fieldAlignedness * (ca.cross(p1 - p0, vec0).T @ (ca.cross(p1 - p0, vec0))) / 2
        crs1 = w_fieldAlignedness * (ca.cross(p1 - p0, vec1).T @ (ca.cross(p1 - p0, vec1))) / 2

        '''

        # --- Energy term 3: curvature alignedness ---
        pc0 = ca.DM(principalCurvatureOnNode[_v0])  # Curvature direction at node _v0
        pc1 = ca.DM(principalCurvatureOnNode[_v1])  # Curvature direction at node _v1
        pc0_ = ca.mtimes(R0.T, pc0)                 # Project to local frame
        pc1_ = ca.mtimes(R1.T, pc1)
        
        dot0 = 0 #w_curvatureAlignedness * 0.5 * ca.power(ca.dot(p1 - p0, pc0_), 2)
        dot1 = 0 #w_curvatureAlignedness * 0.5 * ca.power(ca.dot(p0 - p1, pc1_), 2)
        
        '''
        # Total segment energy (normalized)
        
        energy = (distance_energy + crs0 + crs1) / (edgeLen * totalCurveLength)

        #energy = (distance_energy) / (edgeLen * totalCurveLength)

        #print(type(energy))

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
        #print("p symbolic:", p, "type:", type(p[0]))

        exp = ca.DM(0.5) * ca.DM(p)

        
        distance_energy = ca.power(dx**2 + dy**2 + dz**2, exp)

        #print("distance_energy:", distance_energy, "type:", type(distance_energy))

        # Energy term 2: FIELD‑ALIGNED ENERGY
        vf0 = ca.DM(vectorFieldOnNode[_v0]) if w_fieldAlignedness > tol else ca.DM.zeros(3, 1)  
        vf1 = ca.DM(vectorFieldOnNode[_v1]) if w_fieldAlignedness > tol else ca.DM.zeros(3, 1)  
        vec0 = ca.mtimes(R0.T, vf0)
        vec1 = ca.mtimes(R1.T, vf1)

        crs0 = w_fieldAlignedness * 0.5 * ca.dot(ca.cross(p1 - p0, vec0), ca.cross(p1 - p0, vec0))
        crs1 = w_fieldAlignedness * 0.5 * ca.dot(ca.cross(p0 - p1, vec1), ca.cross(p0 - p1, vec1))

        """ 
        Lets leave this out for now
        # --- Term 3: curvature-alignedness ---
        pc0 = ca.DM(principalCurvatureOnNode[_v0])
        pc1 = ca.DM(principalCurvatureOnNode[_v1])
        pc0_ = ca.mtimes(R0.T, pc0)
        pc1_ = ca.mtimes(R1.T, pc1)

        dot0 = w_curvatureAlignedness * 0.5 * ca.power(ca.dot(p1 - p0, pc0_), 2)
        dot1 = w_curvatureAlignedness * 0.5 * ca.power(ca.dot(p0 - p1, pc1_), 2)
        """ 
        
        # Final energy term for this one-active-node segment
        energy = (distance_energy + crs0 + crs1) / (edgeLen * totalCurveLength)
        #energy = (distance_energy) / (edgeLen * totalCurveLength)

        #print(energy.shape)

        # Append to the global list of energy terms
        energy_terms.append(energy)

    #print('Shortest Curve and Field Alignment is DONE')

    

    ###########################################################################
    # BIHARMONIC ENERGY
    # In the following part we determine the third term of the energy function: 
    # the biHARMONIC energy term.
    # It encourages curve segments to be straighter, rather than squiggly.
    ###########################################################################


    # Initialize lists
    vertexTriplets = []
    rotationMatrices = []
    mappedPoints = []
    mappedLengths = []

    vertexTripletsWith2ActiveNodes = []
    rotationMatricesWith2ActiveNodes = []
    mappedPointsWith2ActiveNodes = []
    mappedLengthsWith2ActiveNodes = []

    vertexTripletsWith1ActiveNode = []
    rotationMatricesWith1ActiveNode = []
    mappedPointsWith1ActiveNode = []
    mappedLengthsWith1ActiveNode = []

    if w_bilaplacian > 0.0:
        # Step 1: Build incident segment list for each node
        node2Segment = [[] for _ in range(len(nodes))]
        for i, (a, b) in enumerate(segments):
            node2Segment[a].append(i)
            node2Segment[b].append(i)

        for i in range(len(nodes)):
            if isFixedNode[i]:
                continue

            incidentSegments = node2Segment[i]
            if len(incidentSegments) != 2:
                continue  # skip boundaries or non-manifold nodes

            s0, s1 = incidentSegments
            v1 = i

            v0 = segments[s0][1] if segments[s0][0] == v1 else segments[s0][0]
            v2 = segments[s1][1] if segments[s1][0] == v1 else segments[s1][0]

            sgn0 = 1 if segments[s0][0] == v1 else -1
            sgn1 = 1 if segments[s1][0] == v1 else -1

            # Extract tangents and bitangents with correct signs
            t0 =  sgn0 * segmentTangent[s0][1] if sgn0 > 0 else -segmentTangent[s0][0]
            b0 =  sgn0 * segmentBitangent[s0][1] if sgn0 > 0 else -segmentBitangent[s0][0]
            n0 = np.cross(t0, b0)

            t1 =  sgn0 * segmentTangent[s0][0] if sgn0 > 0 else -segmentTangent[s0][1]
            b1 =  sgn0 * segmentBitangent[s0][0] if sgn0 > 0 else -segmentBitangent[s0][1]
            n1 = np.cross(t1, b1)

            t2 = -sgn1 * segmentTangent[s1][1] if sgn1 > 0 else segmentTangent[s1][0]
            b2 = -sgn1 * segmentBitangent[s1][1] if sgn1 > 0 else segmentBitangent[s1][0]
            n2 = np.cross(t2, b2)

            # Angle between t1 (v1->v0) and t12 (v1->v2)
            t12 = segmentTangent[s1][0] if sgn1 > 0 else -segmentTangent[s1][1]
            x = np.dot(t1, t12)
            y = np.dot(t12, b1)
            angle = atan2(y, x)

            # Rotation matrices: local frame per segment end
            R0 = np.column_stack((t0, b0, n0))
            R1 = np.column_stack((t1, b1, n1))

            # 2D rotation of final frame by (angle - π)
            rot = angle - pi
            R2_rot = np.array([
                [cos(rot), -sin(rot), 0],
                [sin(rot),  cos(rot), 0],
                [0,         0,        1]
            ])
            R2 = np.column_stack((t2, b2, n2))
            R2 = R2_rot @ R2

            # Segment lengths
            l0 = segmentLengths[s0]
            l1 = segmentLengths[s1]

            # Mapped local points
            p0 = np.array([l0, 0, 0])
            p1 = np.array([0.0, 0.0, 0.0])
            p2 = np.array([l1 * cos(angle), l1 * sin(angle), 0.0])

            # Assert rotation works (optional, dev check)
            # MAYBE THIS ASSERT SHOULD REALLY JUST NOT BE COMMENTED OUT :)
            #assert np.linalg.norm(R2 @ t2 - np.array([-x, -y, 0])) < 1e-6

            # === Classification ===
            is1Active = isFixedNode[v0] and isFixedNode[v2]
            is2Active = isFixedNode[v0] or isFixedNode[v2]

            triplet = [v0, v1, v2]
            rotations = [R0, R1, R2]
            points = [p0, p1, p2]
            lengths = [l0, l1]

            if is1Active:
                vertexTripletsWith1ActiveNode.append(triplet)
                rotationMatricesWith1ActiveNode.append(rotations)
                mappedPointsWith1ActiveNode.append(points)
                mappedLengthsWith1ActiveNode.append(lengths)
            elif is2Active:
                vertexTripletsWith2ActiveNodes.append(triplet)
                rotationMatricesWith2ActiveNodes.append(rotations)
                mappedPointsWith2ActiveNodes.append(points)
                mappedLengthsWith2ActiveNodes.append(lengths)
            else:
                vertexTriplets.append(triplet)
                rotationMatrices.append(rotations)
                mappedPoints.append(points)
                mappedLengths.append(lengths)

    # AGAIN ADD THIS TO THE ENERGY TERMS

    for e_id in range(len(vertexTriplets)):
        _v0, _v1, _v2 = vertexTriplets[e_id]   # Node indices

        # Retrieve symbolic variables for the three nodes
        base = 3 * node2ActiveNodeIdx[_v0]
        x0 = x_sym[base     : base + 3] 

        base = 3 * node2ActiveNodeIdx[_v1]
        x1 = x_sym[base     : base + 3]

        base = 3 * node2ActiveNodeIdx[_v2]
        x2 = x_sym[base     : base + 3]


        # Fixed reference positions (constants)
        v0 = ca.DM(cartesianCoords[_v0])
        v1 = ca.DM(cartesianCoords[_v1])
        v2 = ca.DM(cartesianCoords[_v2])

        # Local rotation matrices
        R0 = ca.DM(rotationMatrices[e_id][0])
        R1 = ca.DM(rotationMatrices[e_id][1])
        R2 = ca.DM(rotationMatrices[e_id][2])

        # Mapped reference points in local frames
        p0_ref = ca.DM(mappedPoints[e_id][0])
        p1_ref = ca.DM(mappedPoints[e_id][1])
        p2_ref = ca.DM(mappedPoints[e_id][2])

        # Transform points to local frame
        p0 = ca.mtimes(R0, x0 - v0) + p0_ref
        p1 = ca.mtimes(R1, x1 - v1) + p1_ref
        p2 = ca.mtimes(R2, x2 - v2) + p2_ref

        # Segment lengths
        l0 = mappedLengths[e_id][0]
        l1 = mappedLengths[e_id][1]

        # First derivatives (segment gradients)
        d0 = (p0 - p1) / l0
        d1 = (p1 - p2) / l1

        # Bilaplacian (second difference)
        d = ca.dot(d0 - d1, d0 - d1)
        l_avg = (l0 + l1) / 2

        # Weighted bilaplacian energy
        energy = (w_bilaplacian * l_avg * d) / totalCurveLength

        #print(energy.shape)

        # Append to energy list
        energy_terms.append(energy)

    #print('-------------Bilaplacian term is DONE--------------')



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
            
            if np.linalg.norm(nodeTangents[v]) < tol:
                nodeTangents[v] -= 0.01 * (n / np.linalg.norm(n))
            if np.linalg.norm(nodeNormals[v]) < tol:
                nodeNormals[v] -= 0.01 * (t / np.linalg.norm(t))
            
            if np.linalg.norm(nodeBitangents[v]) < tol:
                nodeBitangents[v] -= 0.01 * (t / np.linalg.norm(t))

            # Add half the segment length to the node's weight
            nodeWeight[v] += segmentLengths[i] / 2.0

    # Final normalization step for each node's accumulated vectors

    for i in range(num_nodes):

        
        # Tangent
        norm_tan = np.linalg.norm(nodeTangents[i])
        if norm_tan > tol:
            nodeTangents[i] /= norm_tan
        else:
            nodeTangents[i] = np.zeros_like(nodeTangents[i])  # or tol * default_direction

        # Normal
        norm_nml = np.linalg.norm(nodeNormals[i])
        if norm_nml > tol:
            nodeNormals[i] /= norm_nml
        else:
            nodeNormals[i] = np.zeros_like(nodeNormals[i])
        

        # Bitangent
        norm_bit = np.linalg.norm(nodeBitangents[i])
        if norm_bit > tol:
            nodeBitangents[i] /= norm_bit
        else:
            nodeBitangents[i] = np.zeros_like(nodeBitangents[i])

        #print(type(nodeBitangents[i]))
        #print(type(cartesianCoords[i]))

    '''
    # THIS REAALLY NEEDS TO BE REMOVED AFTER DEBUGGING!!!!
    #assert [sp.coord3d for sp in nodes] == cartesianCoords, "Nodes and Cartesian coordinates are no longer consistent"

    print('Is cartesianCoords still conistent with nodes?')
    pairs = list(zip([sp.coord3d for sp in nodes], cartesianCoords))
    if all(np.array_equal(p, q) for p, q in pairs):
        print("All pairs are identical.")
    else:
        print("At least one pair differs.")
    # ------------------------------------------------------------------
    '''

    # Compute medial axis using provided function
    nodeMedialAxis = medial_axis(tri_mesh, nodes, cartesianCoords, 
                                 nodeTangents, nodeNormals, 
                                 nodeBitangents, maxRadius, tracer, 
                                 isGeodesic=False)
    
    #print('These are the medial axis points:', nodeMedialAxis)

    DEBUG = []
    


    print('UPDATEEEDD')

    for j in range(len(nodes)):
        x0 = nodeMedialAxis[j][0]
        x1 = nodeMedialAxis[j][1]
        x2 = np.array([nodes[j].coord3d[0], nodes[j].coord3d[1], nodes[j].coord3d[2]])
        DEBUG.append(np.stack([nodeMedialAxis[j][0], nodeMedialAxis[j][1], nodes[j].coord3d]))
    
    DEBUG = np.stack(DEBUG)
    


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

        #print('The activeNode2NodeIdxhas length', len(activeNode2NodeIdx))
        #print('This is nodeId:', nodeId)

        # Extract medial-axis centers (shape: (3,))
        c0 = ca.DM(nodeMedialAxis[nodeId][0])
        c1 = ca.DM(nodeMedialAxis[nodeId][1])
        #print(f"[e_id={e_id}] c0 type: {type(c0)}, shape: {c0.shape}, c0: {c0.T}")
        #print(f"[e_id={e_id}] c1 type: {type(c1)}, shape: {c1.shape}, c1: {c1.T}")

        # Extract symbolic variable for this node
        base = 3 * e_id # THIS MAYBE WRONG! IT MIGHT HAVE TO BE nodeId

        #print('Just before the error base:',base)
        #print('Just before the error x_sym:',x_sym)
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
        repulsion = (alphas[nodeId] * nodeWeight[nodeId] * d2_sum**(q / 2)) / totalCurveLength
        #print(type(repulsion))
        #print(f"[e_id={e_id}] repulsion symbolic: {repulsion}")

        #print(repulsion.shape)

        energy_terms.append(repulsion)

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


    def make_pos_def(H, delta=1e-6):
        """
        Versucht Cholesky. Wenn es fehlschlägt, wird H <- H + α·I iterativ, 
        bis SPD (symmetric positive definite) erreicht ist.
        """
        H_sym = (H + H.T) / 2  # symmetrisieren
        try:
            # Test auf SPD
            _ = cholesky(H_sym, lower=True, check_finite=True)
            return H_sym
        except LinAlgError:
            # Iterative Verschiebung
            alpha = delta
            while True:
                try:
                    H2 = H_sym + alpha * np.eye(H.shape[0])
                    _ = cholesky(H2, lower=True, check_finite=True)
                    return H2
                except LinAlgError:
                    alpha *= 10


    H_numpy = np.array(H_val.full())
    H_posdef = make_pos_def(H_numpy, delta=1e-6)

    #H_without_proj = np.array(H_val.full())

    #d = np.linalg.solve(H_without_proj, -np.array(g_val).flatten())
    d = np.linalg.solve(H_posdef, -np.array(g_val).flatten())
    g = np.array(g_val).flatten()

    # Map results to descent and gradient fields
    descent = [None] * len(cartesianCoords)
    gradient = [None] * len(cartesianCoords)

    # Store per-node vectors
    for i in range(num_active):
        idx = activeNode2NodeIdx[i]
        base = 3 * i
        descent_vec = d[base : base + 3]
        gradient_vec = g[base : base + 3]
        descent[idx] = np.array(descent_vec)
        gradient[idx] = np.array(gradient_vec)


    #descent = [x if x is not None else np.zeros(3) for x in descent]

    return descent, gradient, ca.Function("E", [x_sym], [E]), nodeMedialAxis, DEBUG
