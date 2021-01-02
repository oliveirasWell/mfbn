#  args, graph: MGraph
# {'vertex': vertex, 'neighborhood': neighborhood, 'twohops': twohops, 'args': args}
def gmb_pure_compute_neigh_list_with_similarity(args, graph_similarity_brodcast):
    twohops = args['twohops']
    vertex = args['vertex']
    current_layer = args['current_layer']

    graph_similarity_function = graph_similarity_brodcast.value

    return list(
        map(
            lambda twohop: (tuple(sorted((vertex, twohop))),
                            {'current_layer': current_layer, 'similarity': graph_similarity_function(vertex, twohop)}
                            ),
            twohops
        )
    )
