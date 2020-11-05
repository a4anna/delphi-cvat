#!/usr/bin/env python3

import signal
import sys
import time
import yaml
from pathlib import Path

from cvat.client import CvatClient
from delphi.utils import yaml_path_matcher, yaml_path_constructor

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


    client = CvatClient(config)
    try:
        client.start()
    except (KeyboardInterrupt, Exception):
        client.stop()
        time.sleep(10)
        sys.exit(1)


if __name__ == '__main__':
    main()

