# TODO Document
import numpy


def gmb_matching_pure(edges: list, vcount: int, reduction_factor: int, vertices):
    merge_count = int(reduction_factor * len(vertices))
    matching = numpy.array([-1] * vcount)
    matching[vertices] = vertices
    visited = [0] * vcount
    for edge, value in edges:
        vertex = edge[0]
        neighbor = edge[1]
        if (visited[vertex] != 1) and (visited[neighbor] != 1):
            matching[neighbor] = vertex
            matching[vertex] = vertex
            visited[neighbor] = 1
            visited[vertex] = 1
            merge_count -= 1
        if merge_count == 0:
            break

    return matching
