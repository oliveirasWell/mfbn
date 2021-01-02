# TODO Document
from typing import List, Dict, Any

import numpy

from models.mgraph import MGraph


def gmb_matching_pure_spark(graph: MGraph, sorted_edges_by_layer: list, broadcast_kwargs: List[Dict[str, Any]]):
    results = []

    for layer in sorted_edges_by_layer:
        layer_number = sorted_edges_by_layer.index(layer)
        broadcast_kwargs_of_layer = broadcast_kwargs[layer_number]

        result = numpy.array([-1] * graph.vcount())
        vertices_ = broadcast_kwargs_of_layer["vertices"]
        result[vertices_] = vertices_
        visited = [0] * graph.vcount()

        merge_count = int(broadcast_kwargs_of_layer["reduction_factor"] * len(vertices_))
        item_list = layer[1]
        item_list_ordered = [i for i in sorted(item_list, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)]

        for edge, value in item_list_ordered:
            vertex = edge[0]
            neighbor = edge[1]
            if (visited[vertex] != 1) and (visited[neighbor] != 1):
                result[neighbor] = vertex
                result[vertex] = vertex
                visited[neighbor] = 1
                visited[vertex] = 1
                merge_count -= 1
            if merge_count == 0:
                break

        results.append(result)

    matching = numpy.arange(graph.vcount())
    for result in results:
        vertices = numpy.where(result > -1)[0]
        matching[vertices] = result[vertices]

    return matching
