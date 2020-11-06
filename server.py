#!/usr/bin/env python3

import os
import signal
import sys
import logging
import traceback
import time
import yaml
from pathlib import Path

from delphi.server import DelphiServer
from delphi.utils import yaml_path_matcher, yaml_path_constructor

logging.basicConfig(level=logging.INFO)
yaml.add_implicit_resolver('!path', yaml_path_matcher, None, yaml.SafeLoader)
yaml.add_constructor('!path', yaml_path_constructor, yaml.SafeLoader)


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
    logging.error("\n".join(code))


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else (Path.cwd() / 'config.yml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config.get('debug', False):
        signal.signal(signal.SIGUSR1, dumpstacks)

    server = DelphiServer(config)

    try:
        server.start()
    except (KeyboardInterrupt, Exception):
        server.stop()
        time.sleep(10)
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    main()

