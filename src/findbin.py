from os.path import dirname, abspath

def findbin(path=None):
    if path is None:
        import sys
        try:
            path = sys.argv[0]
        except IndexError:
            raise Exception('can\'t determine path to script')
    return dirname(abspath(path))
