class SparkCoarseningKwargs:
    def __init__(self, reduction_factor, vertices=None, reverse=None, seed_priority=None, upper_bound=None, n=None,
                 global_min_vertices=None,
                 tolerance=None, itr=None):
        self.reduction_factor = reduction_factor
        self.vertices = vertices
        self.seed_priority = seed_priority
        self.reverse = reverse
        self.upper_bound = upper_bound
        self.n = n
        self.global_min_vertices = global_min_vertices
        self.tolerance = tolerance
        self.itr = itr
        self.merge_count = int(self.reduction_factor * len(self.vertices))

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)
