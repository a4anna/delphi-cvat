class RetrieverStats(object):

    def __init__(self, total_objects: int, dropped_objects: int, false_negatives: int):
        self.total_objects = total_objects
        self.dropped_objects = dropped_objects
        self.false_negatives = false_negatives
