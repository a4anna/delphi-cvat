import sys
import threading
import traceback
from concurrent import futures
from pathlib import Path

import grpc
import multiprocessing_logging
import torchvision
from logzero import logger

from delphi.proto import delphi_pb2_grpc, internal_pb2_grpc, admin_pb2_grpc
from delphi.search_manager import SearchManager
from delphi.servicer.admin_servicer import AdminServicer
from delphi.servicer.internal_servicer import InternalServicer
from delphi.servicer.delphi_servicer import DelphiServicer
from delphi.svm.fs_feature_cache import FSFeatureCache
from delphi.svm.noop_feature_cache import NoopFeatureCache
from delphi.utils import log_exceptions

torchvision.set_image_backend('accimage')

NUM_SEC_WAIT = 2

class DelphiServer(object):

    def __init__(self, config):
        multiprocessing_logging.install_mp_handler()
        port = config['port']
        self.server = grpc.server(futures.ThreadPoolExecutor(), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ])
        manager = SearchManager()
        feature_cache = FSFeatureCache(Path(config['root_dir'])/"features") # NoopFeatureCache()
        self.server.add_insecure_port('0.0.0.0:{}'.format(port))

        delphi_servicer = DelphiServicer(manager, Path(config['root_dir']),
                                         feature_cache, port)
        delphi_pb2_grpc.add_DelphiServiceServicer_to_server(delphi_servicer, self.server)
        internal_pb2_grpc.add_InternalServiceServicer_to_server(InternalServicer(manager), self.server)
        admin_pb2_grpc.add_AdminServiceServicer_to_server(AdminServicer(manager), self.server)

    @log_exceptions
    def start(self):
        logger.info("Server start")
        self.server.start()
        self.server.wait_for_termination()

    def stop(self):
        self.server.stop(NUM_SEC_WAIT)
