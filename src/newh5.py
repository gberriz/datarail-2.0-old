#!/usr/bin/env python
import sys
import os

import h5py

def _warn(msg, prog=None, stream=sys.stderr):
    prefix = '' if prog is None else '%s: ' % prog
    print >> stream, prefix + msg


def _error(*args, **kwargs):
    _warn(*args, **kwargs)
    raise Exception


def newh5(path, noclobber=True):
    if os.path.exists(path):
        if noclobber:
            raise IOError('%s exists' % path)
        else:
            os.remove(path)
    with h5py.File(path, 'w'):
        pass

def main(argv):
    prog = os.path.basename(argv[0])
    try:
        try:
            path = argv[1]
        except IndexError:
            _error('Usage: %s <PATH>' % prog)

        try:    
            newh5(path)
        except Exception, e:
            _error(str(e), prog=prog)

    except:
        return 1

    return 0

if __name__ == '__main__':
    exit(main(sys.argv))
