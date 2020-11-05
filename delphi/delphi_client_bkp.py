import grpc
import sys
import yaml

import delphi.proto.delphi_pb2 as delphi_pb2
from delphi.proto.delphi_pb2_grpc import DelphiServiceStub

def make_message(message):
    return delphi_pb2.Message(
        message=message
    )

def generate_messages():
    messages = [
        make_message("First message"),
        make_message("Second message"),
        make_message("Third message"),
        make_message("Fourth message"),
        make_message("Fifth message"),
    ]
    for msg in messages:
        print("Hello Server Sending you the %s" % msg.message)
        yield msg


def send_message(stub):
    responses = stub.GetMessages(generate_messages())
    for response in responses:
        print("Hello from the server received your %s" % response.message)

def run():

    config_path = sys.argv[1] if len(sys.argv) > 1 else (Path.home() / '.delphi' / 'config.yml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    server_port = config['port']

    channel = grpc.insecure_channel('localhost:{}'.format(server_port))
    stub  = DelphiServiceStub(channel)
    send_message(stub)


if __name__ == '__main__':
    run()