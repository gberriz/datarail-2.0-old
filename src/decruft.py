import sys

REGISTRY = dict()

def register(path, globs):
    global REGISTRY
    REGISTRY[path] = Decrufter(set(globs.keys()))

class Decrufter(object):
    '''
# ... IMPORTS

from decruft import Decrufter
DECRUFTER = Decrufter(globals())

# ... BODY OF MODULE

DECRUFTER.instrument(globals())

if __name__ == '__main__':
    import sys
    DECRUFTER.exit(main(sys.argv)) # any output sent to sys.stderr by default
    '''

    def __init__(self, globs, out=sys.stderr):
        self.out = out
        self.start = set(globs.keys())
        self.ncalls = dict()

    def instrument(self, globs):
        start = self.start
        ncalls = self.ncalls
        for k, v in globs.items():
            if (k in start) or (not hasattr(v, '__call__')):
                continue
            # out = self.out
            # print >> out, 'wrapping %s: %s' % (k, hasattr(v, '__call__')),
            # try:
            #     print >> out, ' (%s)' % v
            # except Exception, e:
            #     print >> out, '*** %s (%s) ***' % (e.__class__.__name__, e)
            globs[k] = self._wrap(k, v)


    def _wrap(self, k, v):
        ncalls = self.ncalls
        ncalls[k] = 0
        def wrapped(*args, **kwargs):
            ncalls[k] += 1
            return v(*args, **kwargs)
        return wrapped


    def report(self):
        ncalls = self.ncalls
        out = self.out
        for k, v in ncalls.items():
            if v == 0:
                print >> out, k

    # Note: 0 is Python's official default return value for
    # __builtins__.exit()
    def exit(self, val=0):
        self.report()
        __builtins__['exit'](val)
