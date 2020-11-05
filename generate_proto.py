#!/usr/bin/env python3

from pathlib import Path
from subprocess import check_call

proto_root = Path.cwd() / 'protos'
proto_dir = proto_root / 'delphi' / 'proto'
for proto_file in proto_dir.iterdir():
    check_call(
        'python -m grpc_tools.protoc -I{} {} --python_out=. --grpc_python_out=. --mypy_out=.'
        .format(proto_root, proto_file)
        .split())

