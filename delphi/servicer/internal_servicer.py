import os
from typing import Iterable, Iterator

import grpc
from google.protobuf.any_pb2 import Any
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import BytesValue
from logzero import logger

from delphi.proto.internal_pb2 import GetExamplesRequest, ExampleMetadata, GetExampleRequest, StageModelRequest, \
    InternalMessage, TrainModelRequest, ValidateTestResultsRequest, SubmitTestRequest, PromoteModelRequest, \
    DiscardModelRequest
from delphi.proto.internal_pb2_grpc import InternalServiceServicer
from delphi.search_manager import SearchManager


class InternalServicer(InternalServiceServicer):

    def __init__(self, manager: SearchManager):
        self._manager = manager

    def GetExamples(self, request: GetExamplesRequest, context: grpc.ServicerContext) -> Iterable[ExampleMetadata]:
        try:
            return self._manager.get_search(request.searchId).get_examples(request.exampleSet, request.nodeIndex)
        except Exception as e:
            logger.exception(e)
            raise e

    def GetExample(self, request: GetExampleRequest, context: grpc.ServicerContext) -> BytesValue:
        try:
            example_path = self._manager.get_search(request.searchId).get_example(request.exampleSet, request.label,
                                                                                  request.key)
            with example_path.open('rb') as f:
                return BytesValue(value=f.read())
        except Exception as e:
            logger.exception(e)
            raise e

    def TrainModel(self, request: TrainModelRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).train_model(request.trainerIndex)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def StageModel(self, request: StageModelRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).stage_model(request.version, request.trainerIndex,
                                                                   request.content)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def ValidateTestResults(self, request: ValidateTestResultsRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).validate_test_results(request.version)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def SubmitTestResults(self, request: Iterator[SubmitTestRequest], context: grpc.ServicerContext) -> Empty:
        try:
            version = next(request).version
            self._manager.get_search(version.searchId).submit_test_results((x.result for x in request), version.version)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def PromoteModel(self, request: PromoteModelRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).promote_model(request.version)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def DiscardModel(self, request: DiscardModelRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).discard_model(request.version)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def CheckBandwidth(self, request: BytesValue, context: grpc.ServicerContext) -> BytesValue:
        try:
            return BytesValue(value=bytearray(os.urandom(1024 * 1024)))
        except Exception as e:
            logger.exception(e)
            raise e

    def MessageInternal(self, request: InternalMessage, context: grpc.ServicerContext) -> Any:
        try:
            return self._manager.get_search(request.searchId).message_internal(request.trainerIndex, request.message)
        except Exception as e:
            logger.exception(e)
            raise e
