from pyspark.accumulators import AccumulatorParam


class ListParam(AccumulatorParam):
    def zero(self, v):
        return v

    def addInPlace(self, acc1, acc2):
        print('acc1')
        print(acc1)
        print('acc2')
        print(acc2)
        vertex = acc2[0]
        neigh = acc2[1]

        # print('vertices')
        # print(acc1)
        # print(acc2)
        # print(vertices)
        # print('vertices')

        acc1[neigh] = vertex
        acc1[vertex] = vertex

        return acc1