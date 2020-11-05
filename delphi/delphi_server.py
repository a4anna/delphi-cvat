import logging
import signal
import sys
import threading
import traceback
from concurrent import futures
from pathlib import Path

import grpc
import logzero
import multiprocessing_logging
import torchvision
import yaml
from logzero import logger

from delphi.proto import delphi_pb2_grpc, internal_pb2_grpc, admin_pb2_grpc
from delphi.search_manager import SearchManager
from delphi.servicer.admin_servicer import AdminServicer
from delphi.servicer.internal_servicer import InternalServicer
from delphi.servicer.delphi_servicer import DelphiServicer
from delphi.svm.fs_feature_cache import FSFeatureCache
from delphi.svm.noop_feature_cache import NoopFeatureCache
from delphi.svm.redis_feature_cache import RedisFeatureCache
from delphi.utils import log_exceptions

logzero.loglevel(logging.INFO)
torchvision.set_image_backend('accimage')


def dumpstacks(_, __):
    traceback.print_stack()
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


@log_exceptions
def main():
    multiprocessing_logging.install_mp_handler()

    config_path = sys.argv[1] if len(sys.argv) > 1 else (Path.home() / '.delphi' / 'config.yml')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get('debug', False):
        signal.signal(signal.SIGUSR1, dumpstacks)

    server = grpc.server(futures.ThreadPoolExecutor(), options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    manager = SearchManager()

    cache_config = config['feature_cache']
    if cache_config['type'] == 'redis':
        feature_cache = RedisFeatureCache(cache_config['port'])
    elif cache_config['type'] == 'filesystem':
        feature_cache = FSFeatureCache(Path(cache_config['feature_dir']))
    elif cache_config['type'] == 'noop':
        feature_cache = NoopFeatureCache()
    else:
        raise NotImplementedError('unknown feature cache type: {}'.format(cache_config))

    port = config['port']

    delphi_servicer = DelphiServicer(manager, Path(config['root_dir']),
                                     feature_cache, port)
    delphi_pb2_grpc.add_DelphiServiceServicer_to_server(delphi_servicer, server)
    internal_pb2_grpc.add_InternalServiceServicer_to_server(InternalServicer(manager), server)
    admin_pb2_grpc.add_AdminServiceServicer_to_server(AdminServicer(manager), server)


    server.add_insecure_port('0.0.0.0:{}'.format(port))
    logger.info('Starting delphi on port: {}'.format(port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
