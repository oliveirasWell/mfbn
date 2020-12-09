def seq_op(acumulator, element):

    if element == -1:
        return acumulator

    vertex, neighbor, similaity = element

    if acumulator[neighbor] != -1 and acumulator[neighbor][2] < similaity:
        actual = acumulator[neighbor]
        acumulator[actual[0]] = -1
        acumulator[actual[1]] = -1

    if acumulator[vertex] != -1 and acumulator[vertex][2] < similaity:
        actual = acumulator[vertex]
        acumulator[actual[0]] = -1
        acumulator[actual[1]] = -1

    if acumulator[neighbor] == -1 and acumulator[vertex] == -1:
        acumulator[neighbor] = (vertex, neighbor, similaity)
        acumulator[vertex] = (vertex, neighbor, similaity)

    return acumulator
