def flat_map_agregated_items(layer):
    return list(
        map(lambda item: (layer[0], -1 if item == -1 else (item[0], item[1], item[2])), layer[1])
    )
