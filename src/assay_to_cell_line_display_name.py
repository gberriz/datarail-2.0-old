from collections import defaultdict

def _new_name(old_name, index):
    return '%s__%s' % (old_name, chr(ord('a') + index))

def make_lookup(assays):
    tally = defaultdict(list)
    for n in assays:
        tally[n[1]].append(n)

    b = ord('a')
    for k in [k for k in tally.keys() if len(tally[k]) > 1]:
        for i, v in enumerate(sorted(tally.pop(k))):
            tally[_new_name(k, i)] = [v]

    return dict(('_'.join(v[0]), k) for k, v in tally.items())


if __name__ == '__main__':
    from glob import glob
    import os.path as op
    import re

    ASSAYSDIR = 'scans/linkfarm/*'
    assays = [op.basename(p).split('_', 1) for p in glob(ASSAYSDIR)]
    lookup = make_lookup(assays)

    DATAPFLEXDIR = ('/home/gfb2/Dropbox/Breast Cancer Ligand Reponse '
                    'Screen/Mario Data/20111025_DataPflex files')

    import csv
    for p in glob(op.join(DATAPFLEXDIR, '*')):
        with open(p, 'r') as fh:
            reader = csv.reader(fh)
            header = reader.next()
            rows = [[lookup[row[0]]] + row[1:] for row in list(reader)]

        b = op.basename(p)
        a = re.sub('_(?:GF|CK).csv', '', b)
        #p = op.join('/tmp/fix', op.basename(p).replace(a, lookup[a]))
        with open(p.replace(a, lookup[a]), 'w') as fh:
            writer = csv.writer(fh, lineterminator='\r\n')
            writer = csv.writer(fh, lineterminator='\r\n')
            writer.writerow(header)
            writer.writerows(rows)
