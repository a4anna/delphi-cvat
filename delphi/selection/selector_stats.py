from typing import Optional


class SelectorStats(object):

    def __init__(self, processed_objects: int, dropped_objects: int, passed_objects: Optional[int],
                 false_negatives: int):
        self.processed_objects = processed_objects
        self.dropped_objects = dropped_objects
        self.passed_objects = passed_objects
        self.false_negatives = false_negatives
