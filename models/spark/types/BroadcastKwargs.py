from models.spark.types.SparkCoarseningArgs import SparkCoarseningArgs


class BroadcastKwargs:
    def __init__(self, value):
        self.value: SparkCoarseningArgs = value
