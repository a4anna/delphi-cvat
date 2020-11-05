import grpc

from delphi.proto.internal_pb2_grpc import InternalServiceStub
from delphi.proto.delphi_pb2_grpc import DelphiServiceStub


class DelphiStub(object):

    def __init__(self, url: str):
        self.url = url

        channel = grpc.insecure_channel(url, options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ])

        self.api = DelphiServiceStub(channel)
        self.internal = InternalServiceStub(channel)
