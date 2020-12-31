from typing import List

from models.mgraph import MGraph
from models.spark.map_vertex_neighborhood import map_vertex_neighborhood
from models.spark.types.SparkCoarseningArgs import SparkCoarseningArgs


def flat_map_two_layers_into_one_list_with_neighborhood(args, broadcastGraph) -> List:
    graph: MGraph = broadcastGraph.value
    spark_args: SparkCoarseningArgs = SparkCoarseningArgs.from_array(args[0])
    vertices = spark_args.kwargs.vertices
    return list(
        map(lambda v: map_vertex_neighborhood(v, graph.neighborhood, graph['adjlist'], spark_args.current_layer),
            vertices)
    )