def _parseargs(argv):
    foo, bar, baz = argv[1:]
    d = dict()
    l = locals()
    params = 'foo bar baz'
    for p in params.split():
        d[p] = l[p]
    _setparams(d)


def _setparams(d):
    global PARAM
    try:
        pd = PARAM.__dict__
        pd.clear()
        pd.update(d)
    except NameError:
        class _param(object): pass
        PARAM = _param()
        _setparams(d)

def main(argv):
    _parseargs(argv)
    return 0


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0], 'FOO', 'BAR', 'BAZ']
    exit(main(sys.argv))
