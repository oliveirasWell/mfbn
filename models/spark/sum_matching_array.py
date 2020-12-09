# TODO Document
import numpy


def sum_matching_array(result, accum_list):
    vertices2 = numpy.where(result > -1)[0]
    accum_list.add((result, vertices2))
