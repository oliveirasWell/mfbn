# ((74, 106), {'current_layer': 1, 'similarity': 1.0}) -> (current_layer, ((74, 106),  similarity)
def gmb_pure_map_reduced(a):
    a_ = (a[-1]['current_layer'], (*a[:-1], a[-1]['similarity']))
    return a_




