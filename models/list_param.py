from pyspark.accumulators import AccumulatorParam


class ListParam(AccumulatorParam):
    def zero(self, v):
        return v

    def addInPlace(self, acc1, acc2):
        vertex = acc2[0]
        neigh = acc2[1]

        acc1[neigh] = vertex
        acc1[vertex] = vertex

        return acc1