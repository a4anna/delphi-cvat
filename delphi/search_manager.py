import threading
from typing import Dict, List, Tuple

from delphi.proto.delphi_pb2 import SearchId
from delphi.search import Search


class SearchAndMetadata(object):

    def __init__(self, search: Search, metadata: str):
        self.search = search
        self.metadata = metadata


class SearchManager(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._searches: Dict[str, SearchAndMetadata] = {}

    def set_search(self, search_id: SearchId, search: Search, metadata: str) -> None:
        with self._lock:
            assert search_id.value not in self._searches
            self._searches[search_id.value] = SearchAndMetadata(search, metadata)

    def get_search(self, search_id: SearchId) -> Search:
        with self._lock:
            return self._searches[search_id.value].search

    def get_searches(self) -> List[Tuple[SearchId, str]]:
        with self._lock:
            return [(SearchId(value=k), v.metadata) for k, v in self._searches.items()]

    def remove_search(self, search_id: SearchId) -> Search:
        with self._lock:
            search = self._searches[search_id.value]
            del self._searches[search_id.value]

        return search.search
