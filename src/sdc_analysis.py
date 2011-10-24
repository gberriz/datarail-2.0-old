import re

class Analysis(object):
    def process_rows(self, rows, index, pass_=0):
        mss = self.methods[pass_]
        for type_, ms in mss.get('table', {}).items():
            for m in ms:
                m(rows, index)

        if not len(mss.get('row', {})):
            return

        for row in rows:
            try:
                self.process_row(row, index, pass_)
            except StopIteration:
                break


    def process_row(self, row, index, pass_=0):
        mss = self.methods[pass_].get('row', {})
        if not len(mss):
            raise StopIteration

        for type_, ms in mss.items():
            for m in ms:
                m(row, index)


    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError('subclass responsibility')


    _METHOD_RE = re.compile(r'^(?:(table)__)?([a-z]+)__(\d+)(?:__.*)?$')

    def __init__(self):
        mr = Analysis._METHOD_RE
        ms = dict()
        for k in sorted(dir(self)):
            try:
                v = getattr(self, k)
            except:
                continue

            if not hasattr(v, '__call__'):
                continue

            m = mr.search(k)
            if not m:
                continue

            pass_ = int(m.group(3))
            level = m.group(1) or 'row'
            type_ = m.group(2)

            ms.setdefault(pass_, dict()).\
                                 setdefault(level, dict()).\
                                 setdefault(type_, []).append(v)

        self.methods = ms


    @property
    def passes(self):
        return sorted(self.methods.keys())


    process_row = \
    finish = \
    report = \
                _not_implemented
