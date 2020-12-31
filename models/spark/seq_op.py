def seq_op(accumulator, element):

    if element == -1:
        return accumulator

    vertex, neighbor, similaity = element

    if accumulator[neighbor] != -1 and accumulator[neighbor][2] < similaity:
        actual = accumulator[neighbor]
        accumulator[actual[0]] = -1
        accumulator[actual[1]] = -1

    if accumulator[vertex] != -1 and accumulator[vertex][2] < similaity:
        actual = accumulator[vertex]
        accumulator[actual[0]] = -1
        accumulator[actual[1]] = -1

    if accumulator[neighbor] == -1 and accumulator[vertex] == -1:
        accumulator[neighbor] = (vertex, neighbor, similaity)
        accumulator[vertex] = (vertex, neighbor, similaity)

    return accumulator
