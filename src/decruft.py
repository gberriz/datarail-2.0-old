REGISTRY = dict()

def register(path, globs):
    global REGISTRY
    REGISTRY[path] = Decrufter(set(globs.keys()))

class Decrufter(object):
    def __init__(self, globs):
        self.start = set(globs.keys())
        self.ncalls = dict()

    def instrument(self, globs):
        start = self.start
        ncalls = self.ncalls
        for k, v in globs.items():
            if (k in start) or (not hasattr(v, '__call__')):
                continue
            # print 'wrapping %s: %s' % (k, hasattr(v, '__call__')),
            # try:
            #     print ' (%s)' % v
            # except Exception, e:
            #     print '*** %s (%s) ***' % (e.__class__.__name__, e)
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
        for k, v in ncalls.items():
            if v == 0:
                print k

