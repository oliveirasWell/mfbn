def comb_op(acumulator, acumulator2):
    for element in acumulator2:

        if element == -1:
            continue

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
            acumulator[vertex] = (vertex, neighbor,   similaity)

    return acumulator
