import glob
import os
from typing import Optional, Iterable
from logzero import logger
import threading

from delphi.attribute_provider import SimpleAttributeProvider
from delphi.object_provider import ObjectProvider
from delphi.proto.delphi_pb2 import DirectoryDataset
from delphi.retrieval.retriever import Retriever


class DirectoryRetriever(Retriever):

    def __init__(self, dataset: DirectoryDataset):
        self._dataset = dataset
        loop = self._dataset.loop
        self._loop = loop if loop > 0 else 1
        self._start_event = threading.Event()
        self._final_stats = None
        self._obj_retriever = self.get_directory()

    def _fetch_files(self):
        extensions = ('*.jpg', '*.png', '*.jpeg')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self._dataset.name, ext)))
        assert len(image_files), "Image Folder Empty"
        return image_files

    def start(self) -> None:
        self._start_event.set()

    def stop(self) -> None:
        self._start_event.clear()

    def get_directory(self) -> Iterable[ObjectProvider]:
        for loop in range(self._loop):
            logger.info("Dataset {} Loop Number {}".format(self._dataset.name, loop))
            paths = self._fetch_files()
            for path in paths:
                content = b''
                with open(path, "rb") as f:
                    content = f.read()
                yield ObjectProvider(path, content, SimpleAttributeProvider({}), False)
        logger.info("Dataset Completed!")

    def get_objects(self) -> Iterable[ObjectProvider]:
        for loop in range(self._loop):
            logger.debug("Dataset {} Loop Number {}".format(self._dataset.name, loop))
            paths = self._fetch_files()
            for path in paths:
                content = b''
                with open(path, "rb") as f:
                    content = f.read()
                yield ObjectProvider(path, content, SimpleAttributeProvider({}), False)
        logger.info("Dataset Completed!")
        # try:
        #     while True:
        #         yield next(self._obj_retriever)
        # except StopIteration:
        #     yield ObjectProvider('', b'', SimpleAttributeProvider({}), False)
