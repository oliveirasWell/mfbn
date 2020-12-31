# TODO Document
from typing import List, Dict, Any

import numpy

from models.mgraph import MGraph


def gmb_matching_pure_spark(graph: MGraph, sorted_edges_by_layer: list, broadcast_kwargs: List[Dict[str, Any]], vertices):
    final_matching = numpy.arange(graph.vcount())
    result = numpy.array([-1] * graph.vcount())
    result[vertices] = vertices

    visited = [0] * graph.vcount()

    for item in sorted_edges_by_layer:
        layer_number = sorted_edges_by_layer.index(item)
        broadcast_kwargs_of_layer = broadcast_kwargs[layer_number]

        merge_count = int(broadcast_kwargs_of_layer["reduction_factor"] * len(broadcast_kwargs_of_layer["vertices"]))

        item_list = item[1]
        for i in item_list:
            vertex, neighbor = i[0]
            if (visited[vertex] != 1) and (visited[neighbor] != 1):
                result[neighbor] = vertex
                result[vertex] = vertex
                visited[neighbor] = 1
                visited[vertex] = 1
                merge_count -= 1
            if merge_count == 0:
                break

    vertices = numpy.where(result > -1)[0]
    final_matching[vertices] = result[vertices]
    return final_matching
