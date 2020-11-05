import json
from typing import List, Optional, Dict

import redis

from delphi.svm.feature_cache import FeatureCache


class RedisFeatureCache(FeatureCache):

    def __init__(self, port: int):
        self._db = redis.StrictRedis(port=port)

    def get(self, key: str) -> Optional[List[float]]:
        result = self._db.get(key)
        if result is None:
            return None

        return json.loads(result)

    def put(self, values: Dict[str, List[float]], expire: bool) -> None:
        to_cache = dict()
        for key in values:
            to_cache[key] = json.dumps(values[key])

        self._db.mset(to_cache)

        if expire:
            for key in values:
                self._db.expire(key, 600)
