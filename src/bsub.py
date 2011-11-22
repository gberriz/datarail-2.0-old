import subprocess
import os
import re

def is_accessible_queue(queue):
    user = str(os.getuid())
    cmd = ('bqueues', '-u', user, queue)
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, err = p.communicate()

    if err:
        if ('%s: User cannot use the queue' % user) in err:
            return False
        raise OSError(err)
            
    if queue not in out:
        raise Exception('unknown error')

    return True


def bsub(args, queue=None,
         _verbose=False,
         _re=re.compile('Job <(\d+)> is submitted to ')):

    cmd = ['bsub']
    if queue is None:
        try:
            queue = args[args.index('-q') + 1]
        except Exception, e:
            raise
    else:
        cmd += ['-q', queue]

    assert is_accessible_queue(queue)

    p = subprocess.Popen(tuple(cmd) + tuple(args),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate()
    if err:
        raise OSError(err)

    m = _re.search(out)

    if not m:
        raise Exception('unknown error: %s' % out)
    jobid = m.group(1)

    if _verbose:
        print out

    return jobid
