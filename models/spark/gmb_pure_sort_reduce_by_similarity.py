def gmb_pure_map_neight_with_great_similarity(a, b):
    return a if a['similarity'] > b['similarity'] else b
