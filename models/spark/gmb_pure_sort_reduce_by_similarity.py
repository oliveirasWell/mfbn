def gmb_pure_sort_reduce_by_similarity(a, b):
    return a if a['similarity'] > b['similarity'] else b
