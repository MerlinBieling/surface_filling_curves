import math
import numpy as np

from surface_filling_energy_geodesic_mod import surface_filling_energy_geodesic
from surface_path_evolution_mod import surface_path_evolution

def single_iteration(
    nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode,
    tri_mesh, meshlib_mesh, solver, tracer, dictionary,
    vectorField,
    maxRadius, radius, h, 
    useAnisotropicAlphaOnNodes, useAnisotropicAlphaOnMesh,
    alphaRatioOnNodes, alphaRatioOnMesh,
    w_fieldAlignedness, w_curvatureAlignedness, w_biharmonic,
    p, q
):
    
    tol=1e-6
    cartesianCoords = [sp.coord3d for sp in nodes]

    #if h <= tol:
    #    h = math.pi * radius / 25 # h is the boundary value: segments with length < h need to be "collapsed"

    h = math.pi * radius / 25 # h is the boundary value: segments with length < h need to be "collapsed"

    if maxRadius <= tol:
        maxRadius = radius * 10

    
    d, g, f, medialAxis, DEBUG = surface_filling_energy_geodesic(
        nodes,
        cartesianCoords,
        segments,
        segmentSurfacePoints,
        segmentLengths,
        isFixedNode,
        tri_mesh,
        tracer,
        maxRadius,
        radius,
        useAnisotropicAlphaOnNodes,
        useAnisotropicAlphaOnMesh,
        alphaRatioOnNodes,
        alphaRatioOnMesh,
        vectorField,
        w_fieldAlignedness,
        w_curvatureAlignedness,
        w_biharmonic,
        p,
        q
    )
    
    

    new_nodes, new_segments, new_segmentSurfacePoints, new_segmentLengths, new_isFixedNode, retractionPaths = surface_path_evolution(
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
        solver, 
        dictionary
    )
    

    
    return new_nodes, new_segments, new_segmentSurfacePoints, new_segmentLengths, new_isFixedNode, retractionPaths, d, g, f, medialAxis, DEBUG

