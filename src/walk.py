import os
from os.path import join as opjoin

def walk(top, pre=None, post=None, carry=None,
         prune=lambda x: False, onerror=None, followlinks=False):

    """Traverse a directory tree, executing pre-order and post-order
    callbacks.

    Despite its name, the default os.walk is not a true directory tree
    walker, but rather an iterator of tree nodes.  Although this
    iterator can be used to mimic a tree walker, it cannot support
    both a pre-order and post-order action in the same traversal.

    This function, in contrast, is a true tree walker, not an iterator
    (in fact, it returns nothing), taking pre-order and post-order
    callbacks among its arguments.

    The arguments to this function are:

      top:     the directory at the root of the tree to be traversed;
      pre:     callback called before visiting the subdirectories of
               top (see below); defaults to None;
      post:    callback called after visiting the subdirectories of
               top (see below); defaults to None;
      carry:   an arbitrary container object to be passed in every
               invocation of the pre and post callbacks; defaults to
               None;
      prune:   a callback that gets called immediately upon visiting
               the top directory, with the top parameter as its only
               argument; if its returned value has boolean value True,
               the traversal below the top directory is terminated;
               the default value of this callback always returns False;
      onerror: a callback called if an exception occurs while getting
               the contents of the top directory; it receives the
               exception as its sole argument; the traversal below the
               current (top) directory is subsequently terminated; in
               this case, neither the pre nor the post callbacks are
               called; defaults to None;
      followlinks: if True, directories that are symbolic links are
               not visited; defaults to False;

    Both pre and post callbacks receive these four arguments:

      curdir:  the path to the node being visited currently;
      dirs:    a list of the basenames of the subdirectories
               immediately below curdir; the contents of this list
               determine the subsequent traversal below curdir;
               therefore, the pre callback can be used to steer this
               traversal, by modifying the dirs list in-place (this is
               the same scheme as that used by the os.walk function in
               the standard library uses);
      nondirs: a list of the basenames of the non-directory children
               of curdir;
      carry:   whatever the value of the carry parameter of the walk
               function (see description above); this can be, e.g. a
               list to collect items found during the traversal, etc.
    """
    # FIXME: the prune callback is redundant, since the same
    # functionality can be achieved through the pre callback

    if bool(prune(top)):
        return

    if pre is None and post is None:
        raise TypeError('at least one of pre and post must be a callable')

    islink, isdir = os.path.islink, os.path.isdir

    # We may not have read permission for top, in which case we can't
    # get a list of the files the directory contains.  os.path.walk
    # always suppressed the exception then, rather than blow up for a
    # minor reason when (say) a thousand readable directories are still
    # left to visit.  That logic is copied here.
    try:
        names = os.listdir(top)
    except os.error, err:
        if onerror is not None:
            onerror(err)
        return

    dirs, nondirs = [], []
    for name in names:
        p = opjoin(top, name)
        if prune(p):
            continue
        elif os.path.isdir(p):
            dirs.append(name)
        else:
            nondirs.append(name)

    dirs.sort()
    nondirs.sort()

    if pre:
        pre(top, dirs, nondirs, carry)

    for newtop in [opjoin(top, d) for d in dirs]:
        if followlinks or not os.path.islink(newtop):
            walk(newtop, pre, post, carry, prune, onerror, followlinks)

    if post:
        post(top, dirs, nondirs, carry)

def find(root, wanted=None):
    if not os.path.isdir(root):
        raise ValueError('not a directory: %s' % root)
    for r, ds, fs in os.walk(root):
        ds.sort()
        fs.sort()
        for s in (ds, fs):
            for i in s:
                if wanted is None or wanted(i, r, isdir=(s==ds)):
                    yield opjoin(r, i)
