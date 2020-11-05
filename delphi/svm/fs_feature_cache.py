import pickle
from pathlib import Path
from typing import List, Optional, Dict

from logzero import logger

from delphi.svm.feature_cache import FeatureCache


class FSFeatureCache(FeatureCache):

    def __init__(self, feature_dir: Path):
        self._feature_dir = feature_dir
        feature_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[List[float]]:
        feature_path = self._feature_dir / key
        if not feature_path.exists():
            return None

        try:
            with feature_path.open('rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.exception(e)
            feature_path.unlink()
            return None

    def put(self, values: Dict[str, List[float]], expire: bool) -> None:
        for key in values:
            feature_path = self._feature_dir / key
            with feature_path.open('wb') as f:
                pickle.dump(values[key], f)
