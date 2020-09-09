from models.spark.types.SparkCoarseningKwargs import SparkCoarseningKwargs

from types import *


class SparkCoarseningArgs:
    def __init__(self, matching_function_spark: FunctionType, kwargs: dict, current_layer: int):
        self.current_layer = current_layer
        self.kwargs = SparkCoarseningKwargs(**dict(kwargs))
        self.matching_function_spark = matching_function_spark

    def __getitem__(self, key):
        return getattr(self, key)

    @staticmethod
    def from_array(array):
        return SparkCoarseningArgs(array[0], array[1], array[2])
