import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import Int32Value
from logzero import logger

from delphi.proto.admin_pb2 import ResetRequest
from delphi.proto.admin_pb2_grpc import AdminServiceServicer
from delphi.proto.delphi_pb2 import SearchId
from delphi.search_manager import SearchManager


class AdminServicer(AdminServiceServicer):

    def __init__(self, manager: SearchManager):
        self._manager = manager

    def Reset(self, request: ResetRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).reset(request.trainOnly)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def GetLastTrainedVersion(self, request: SearchId, context: grpc.ServicerContext) -> Int32Value:
        try:
            return Int32Value(value=self._manager.get_search(request).get_last_trained_version())
        except Exception as e:
            logger.exception(e)
            raise e