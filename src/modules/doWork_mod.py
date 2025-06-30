from surface_filling_energy_geodesic_mod import surface_filling_energy_geodesic
from surface_path_evolution_mod import surface_path_evolution
from check_intersection_mod import check_intersection
import math

def doWork(
    nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode,
    tri_mesh, meshlib_mesh, solver, tracer,
    vectorField,
    maxRadius, radius,
    useAnisotropicAlphaOnNodes, useAnisotropicAlphaOnMesh,
    alphaRatioOnNodes, alphaRatioOnMesh,
    w_fieldAlignedness, w_bilaplacian,
    p, q
):
    
    cartesianCoords = [sp.coord3d for sp in nodes]
    
    h = math.pi * radius / 25 # h is the boundary value: segments with length < h need to be "collapsed"

    d, g, f, medialAxis = surface_filling_energy_geodesic(
        nodes,
        cartesianCoords,
        segments,
        segmentSurfacePoints,
        segmentLengths,
        isFixedNode,
        tri_mesh,
        maxRadius,
        radius,
        useAnisotropicAlphaOnNodes,
        useAnisotropicAlphaOnMesh,
        alphaRatioOnNodes,
        alphaRatioOnMesh,
        vectorField,
        w_fieldAlignedness,
        w_bilaplacian, #change this name to biharmonic
        p,
        q
    )

    #print ('This is the direction:',d)

    '''
    nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode, retractionPaths = surface_path_evolution(
        tri_mesh,
        meshlib_mesh,
        nodes,
        segments,
        segmentSurfacePoints,
        segmentLengths,
        isFixedNode,
        h,
        d,
        tracer,
        solver
    )

    
    return {
        "nodes": nodes,
        "segments": segments,
        "segmentSurfacePoints": segmentSurfacePoints,
        "segmentLengths": segmentLengths,
        "isFixedNode": isFixedNode,
        "retractionPaths": retractionPaths,
        "descentDirection": d,
        "gradient": g,
        "energy": f,
        "medialAxis": medialAxis
    }
    '''

    return d, g, f, medialAxis
