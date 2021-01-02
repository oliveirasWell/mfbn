from typing import List

from models.mgraph import MGraph
from models.spark.functions.map_vertex_neighborhood import map_vertex_neighborhood
from models.spark.types.SparkCoarseningArgs import SparkCoarseningArgs


def gmb_pure_flat_map_two_layers_into_one_list_with_neighborhood(args, broadcast_graph) -> List:
    graph: MGraph = broadcast_graph.value
    spark_args: SparkCoarseningArgs = SparkCoarseningArgs.from_array(args[0])
    vertices = spark_args.kwargs.vertices

    return list(
        map(lambda v: map_vertex_neighborhood(v, graph.neighborhood, graph['adjlist'], spark_args.current_layer),
            vertices)
    )