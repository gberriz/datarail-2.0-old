# adapted from http://code.activestate.com/recipes/576694-orderedset on
# 110302W

# TODO: implement OrderedSet.index

import collections

KEY, PREV, NEXT = range(3)

class OrderedSet(collections.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next_]
        if iterable is not None:
            self |= iterable


    def __len__(self):
        return len(self.map)


    def __contains__(self, key):
        return key in self.map


    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[PREV]
            curr[NEXT] = end[PREV] = self.map[key] = [key, curr, end]


    def clear(self):
        end = self.end
        map_ = self.map
        curr = end[NEXT]
        while True:
            key, _, next_ = curr
            while curr:
                curr.pop()
            if curr is end:
                break
            map_.pop(key)
            curr = next_


    def copy(self):
        return type(self)(self)


    def difference(self, other):
        return type(self)(filter(lambda x: not x in other, self))


    def difference_update(self, other):
        for v in other:
            self.discard(v)
                                 

    def discard(self, key):
        try:
            self.remove(key)
        except KeyError:
            pass


    def intersection(self, other):
        return type(self)(filter(lambda x: x in other, self))


    __and__ = intersection


    def intersection_update(self, other):
        for v in self:
            if not v in other:
                self.discard(v)


    __iand__ = intersection_update


    def isdisjoint(self, other):
        short, long_ = (self, other) if len(self) < len(other) else (other, self)
        for item in short:
            if item in long_:
                return False
        return True


    def issubset(self, other):
        return all([o in other for o in self])


    def issuperset(self, other):
        return all([o in self for o in other])


    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = next(reversed(self)) if last else next(iter(self))
        self.remove(key)
        return key


    def remove(self, key):
        key, prev, next_ = self.map.pop(key)
        prev[NEXT] = next_
        next_[PREV] = prev


    def symmetric_difference(self, other):
        ret = self.difference(other)
        if not hasattr(other, 'difference'):
            other = type(self)(other)
        return ret.union(other.difference(self))


    __xor__ = symmetric_difference


    def symmetric_difference_update(self, other):
        self.difference_update(other)
        if not hasattr(other, 'difference'):
            other = type(self)(other)
        self.update(other.difference(self))


    __ixor__ = symmetric_difference_update


    def union(self, *others):
        ret = self.copy()
        ret.update(*others)
        return ret


    def __or__(self, other):
        return self.union(other)


    def __ror__(self, other):
        return OrderedSet(other).union(self)

    __add__ = __or__

    __radd__ = __ror__

    def update(self, *others):
        for other in others:
            for item in other:
                self.add(item)


    __ior__ = update


    def __iter__(self):
        end = curr = self.end
        while True:
            curr = curr[NEXT]
            if curr is end:
                break
            yield curr[KEY]

        # curr = end[NEXT]
        # while curr is not end:
        #     yield curr[KEY]
        #     curr = curr[NEXT]


    def __reversed__(self):
        end = curr = self.end
        while True:
            curr = curr[PREV]
            if curr is end:
                break
            yield curr[KEY]


    # def __reversed__(self):
    #     end = self.end
    #     curr = end[PREV]
    #     while curr is not end:
    #         yield curr[KEY]
    #         curr = curr[PREV]


    def __repr__(self):
        if not self:
            return '%s()' % (type(self).__name__,)
        return '%s(%r)' % (type(self).__name__, list(self))


    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


    def __del__(self):
        self.clear()                    # remove circular references




if __name__ == '__main__':
    print(OrderedSet('abracadaba'))
    print(OrderedSet('simsalabim'))
