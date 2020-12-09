import numpy

from models.mgraph import MGraph


def contract_pure(input_graph: MGraph, matching: numpy.ndarray):
    """
    Create coarse graph from matching of groups
    """

    # Contract vertices: Referencing the original graph of the coarse graph

    print("matching of iteration")
    print(len(matching))
    print(len(list(set(matching))))
    print([i for i in matching])

    types = []
    weights = []
    sources = []
    predecessors = []
    matching = numpy.array(matching)
    uniqid = 0
    clusters = numpy.unique(matching)
    for cluster_id in clusters:
        vertices = numpy.where(matching == cluster_id)[0]
        weight = 0
        if len(vertices) > 0:
            source = []
            predecessor = []
            for vertex in vertices:
                input_graph.vs[vertex]['successor'] = uniqid
                weight += input_graph.vs[vertex]['weight']
                source.extend(input_graph.vs[vertex]['source'])
                predecessor.append(vertex)
            weights.append(weight)
            types.append(input_graph.vs[vertices[0]]['type'])
            sources.append(source)
            predecessors.append(predecessor)
            uniqid += 1

    # Create coarsened version
    coarse = MGraph()
    coarse.add_vertices(uniqid)
    coarse.vs['type'] = types
    coarse.vs['weight'] = weights
    coarse.vs['name'] = range(coarse.vcount())
    coarse.vs['successor'] = [None] * coarse.vcount()
    coarse.vs['source'] = sources
    coarse.vs['predecessor'] = predecessors
    coarse['layers'] = input_graph['layers']
    coarse['similarity'] = None
    coarse['vertices'] = []

    coarse['vertices_by_type'] = []
    for layer in range(input_graph['layers']):
        coarse['vertices_by_type'].append(coarse.vs.select(type=layer).indices)
        coarse['vertices'].append(len(coarse['vertices_by_type'][layer]))

    # Contract edges
    dict_edges = dict()
    for edge in input_graph.es():
        v_successor = input_graph.vs[edge.tuple[0]]['successor']
        u_successor = input_graph.vs[edge.tuple[1]]['successor']

        # Add edge in coarsened graph
        if v_successor < u_successor:
            dict_edges[(v_successor, u_successor)] = dict_edges.get((v_successor, u_successor), 0) + edge['weight']
        else:
            dict_edges[(u_successor, v_successor)] = dict_edges.get((u_successor, v_successor), 0) + edge['weight']

    if len(dict_edges) > 0:
        edges, weights = list(zip(*dict_edges.items()))
        coarse.add_edges(edges)
        coarse.es['weight'] = weights
        coarse['adjlist'] = list(map(set, coarse.get_adjlist()))

    return coarse