from models.spark.types.SparkCoarseningKwargs import SparkCoarseningKwargs


def refine_args_func(b):
    args_refined = SparkCoarseningKwargs(**b)
    args_refined.vertices = None
    return args_refined