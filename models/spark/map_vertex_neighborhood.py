from typing import Dict


def map_vertex_neighborhood(vertex, neighborhood, adjlist, current_layer) -> Dict:
    neighborhood = neighborhood(vertices=vertex, order=2)
    twohops = neighborhood[(len(adjlist[vertex]) + 1):]
    return {'vertex': vertex,  'twohops': twohops, 'current_layer': current_layer}