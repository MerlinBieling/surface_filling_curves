

import numpy as np
import trimesh
import copy

from single_iteration_mod import single_iteration


def generate_surface_filling_curve( n_iterations, nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode,
            tri_mesh, meshlib_mesh, solver, tracer, dictionary,
            vector_field,
            maxRadius, radius, h,
            useAnisotropicAlphaOnNodes, useAnisotropicAlphaOnMesh,
            alphaRatioOnNodes, alphaRatioOnMesh,
            w_fieldAlignedness, w_curvatureAlignedness, w_biharmonic,
            p, q
        ):
    
    #print('generate_surface_filling_curve can be accessed')
    
    data = {
    "iteration": 0,
    "nodes": nodes,
    "segments": segments,
    "segmentSurfacePoints": segmentSurfacePoints,
    "segmentLengths": segmentLengths,
    "descent": None,
    "medialAxis": None,
    "DEBUG": None
    }

    curve_iterations = [data]

    #print(len(isFixedNode), len(nodes))
    
    for i in range(n_iterations):    

        #print('This is iteration:', i)
        
        new_nodes = copy.deepcopy(nodes)
        new_segments = copy.deepcopy(segments)
        new_segmentSurfacePoints = copy.deepcopy(segmentSurfacePoints)
        new_segmentLengths = copy.deepcopy(segmentLengths)
        new_isFixedNode = copy.deepcopy(isFixedNode)
        '''
        new_nodes = nodes
        new_segments = segments
        new_segmentSurfacePoints = segmentSurfacePoints
        new_segmentLengths = segmentLengths
        new_isFixedNode = isFixedNode
        '''
        nodes, segments, segmentSurfacePoints, segmentLengths, isFixedNode, retractionPaths, d, g, f, medialAxis, DEBUG = single_iteration(
            new_nodes, new_segments, new_segmentSurfacePoints, new_segmentLengths, new_isFixedNode,
            tri_mesh, meshlib_mesh, solver, tracer, dictionary,
            vector_field,
            maxRadius, radius, h,
            useAnisotropicAlphaOnNodes, useAnisotropicAlphaOnMesh,
            alphaRatioOnNodes, alphaRatioOnMesh,
            w_fieldAlignedness, w_curvatureAlignedness, w_biharmonic,
            p, q
        )


        medialAxis = np.stack(medialAxis)


        data = {
            "iteration": i,
            "nodes": nodes,
            "segments": segments,
            "segmentSurfacePoints": segmentSurfacePoints,
            "segmentLengths": segmentLengths,
            "descent": d,
            "medialAxis": medialAxis,
            "DEBUG": DEBUG
        }

        curve_iterations.append(data)
    

    return curve_iterations






