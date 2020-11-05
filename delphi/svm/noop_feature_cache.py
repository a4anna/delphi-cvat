from typing import List, Optional, Dict

from delphi.svm.feature_cache import FeatureCache


class NoopFeatureCache(FeatureCache):

    def get(self, key: str) -> Optional[List[float]]:
        return None

    def put(self, values: Dict[str, List[float]], expire: bool) -> None:
        pass
