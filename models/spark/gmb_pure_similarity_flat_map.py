#  args, graph: MGraph
# {'vertex': vertex, 'neighborhood': neighborhood, 'twohops': twohops, 'args': args}
def gmb_pure_similarity_flat_map(args, graph_similarity_brodcast):
    # graph: MGraph = broadcastGraph.value
    twohops = args['twohops']
    vertex = args['vertex']
    current_layer = args['current_layer']

    # graph_similarity = graph['similarity']
    graph_similarity_function = graph_similarity_brodcast.value

    return list(
        map(
            lambda twohop: (tuple(sorted((vertex, twohop))),  # 169 181
                            # lambda twohop: (tuple((vertex, twohop)),  # 152 161  # 169 181
                            {'current_layer': current_layer,
                             'similarity': graph_similarity_function(vertex, twohop)}),
            twohops
        )
    )
