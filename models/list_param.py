from pyspark.accumulators import AccumulatorParam


class ListParam(AccumulatorParam):
    def zero(self, v):
        return v

    def addInPlace(self, acc1, acc2):
        results = acc2[0]
        vertices = acc2[1]

        # print('vertices')
        # print(acc1)
        # print(acc2)
        # print(vertices)
        # print('vertices')

        if hasattr(vertices, "__len__"):
            for vertex in vertices:
                acc1[vertex] = results[vertex]

        return acc1