import os.path as op

with open(op.join(op.dirname(__file__), 'VERSION'), 'rt') as f:
    __version__ = f.read().strip()
