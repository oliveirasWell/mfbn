import operator
from typing import List, Dict

import numpy

from models.mgraph import MGraph
from models.spark.types.SparkCoarseningArgs import SparkCoarseningArgs


def gmb_pure_flat(args, broadcastGraph) -> List:
    graph: MGraph = broadcastGraph.value
    spark_args: SparkCoarseningArgs = SparkCoarseningArgs.from_array(args[0])
    vertices = spark_args.kwargs.vertices
    return list(map(lambda v: map_vertex_neighborhood(v, graph, spark_args.current_layer), vertices))


def map_vertex_neighborhood(vertex, graph: MGraph, current_layer) -> Dict:
    neighborhood = graph.neighborhood(vertices=vertex, order=2)
    twohops = neighborhood[(len(graph['adjlist'][vertex]) + 1):]
    return {'vertex': vertex, 'neighborhood': neighborhood, 'twohops': twohops, 'current_layer': current_layer}


#  args, graph: MGraph
# {'vertex': vertex, 'neighborhood': neighborhood, 'twohops': twohops, 'args': args}
def gmb_pure_similarity_flat_map(args, graph_similarity_brodcast):
    # graph: MGraph = broadcastGraph.value
    twohops = args['twohops']
    vertex = args['vertex']
    current_layer = args['current_layer']

    # graph_similarity = graph['similarity']
    graph_similarity = graph_similarity_brodcast.value

    l = list(
        map(
            lambda twohop: (tuple(sorted((vertex, twohop))),  # 169 181
                            # lambda twohop: (tuple((vertex, twohop)),  # 152 161  # 169 181
                            {'current_layer': current_layer,
                             'similarity': graph_similarity(vertex, twohop)}),
            twohops
        )
    )

    if 781 == vertex:
        print("vertex ------- ")
        print(vertex)
        print(l)
        print("vertex ------- ")

    return l


# edges = sorted(dict_edges.items(), key=operator.itemgetter(1), reverse=reverse)
# return {'edges': edges, 'args': args, 'layer': current_layer}


def gmb_pure(args, graph: MGraph):
    spark_args = SparkCoarseningArgs.from_array(args[0])
    kwargs = spark_args.kwargs
    vertices = kwargs.vertices
    reverse = kwargs.reverse
    current_layer = spark_args.current_layer

    vcount = graph.vcount()
    matching = numpy.array([-1] * vcount)
    matching[vertices] = vertices

    # Search two-hopes neighborhood for each vertex in selected layer
    dict_edges = dict()
    visited = [0] * vcount

    for vertex in vertices:
        neighborhood = graph.neighborhood(vertices=vertex, order=2)
        twohops = neighborhood[(len(graph['adjlist'][vertex]) + 1):]

        for twohop in twohops:
            if visited[twohop] == 1:
                continue
            dict_edges[(vertex, twohop)] = graph['similarity'](vertex, twohop)
        visited[vertex] = 1

    edges = sorted(dict_edges.items(), key=operator.itemgetter(1), reverse=reverse)
    return {'edges': edges, 'vcount': vcount, 'args': args, 'layer': current_layer}
