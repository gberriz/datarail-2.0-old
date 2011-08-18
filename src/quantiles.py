from math import modf

# class _hatQ_8(object):
#     def __init__(self, data):
#         self.n = len(data)
#         self.sortedx = sorted(data)

#         _den = (1.0 + 3.0 * n)
#         _f = 1.0/_den
#         p_ = f * 2.0
#         _p = f * (_den - 5.0)
        
#         self.clipp = lambda p: max(p_, min(p, p_))


def quantiles(x, probs):
    # Behaves similarly to R's
    # quantile(x, probs=probs, type=8)

    # > quantile(seq(3), probs = c(0.1 * (seq(10) - 1), 1.0), type = 8)
    #       0%      10%      20%      30%      40%      50%      60%
    # 1.000000 1.000000 1.000000 1.333333 1.666667 2.000000 2.333333
    #      70%      80%      90%     100%
    # 2.666667 3.000000 3.000000 3.000000
    #
    # >>> quantiles(range(1, 4), [0.1 * i for i in range(10)] + [1.0])           
    # [1, 1, 1, 1.3333333333333335, 1.6666666666666667, 2,
    #  2.3333333333333335, 2.666666666666667, 3, 3, 3]

    n = len(x)
    if n == 0:
        return None

    sortedx = sorted(x)

    class _hatQ_8Err(Exception): pass

    def _hatQ_8(p):
        # Implements \hat{Q}_8 from
        # Hyndman, R. J. and Fan, Y. (1996) Sample quantiles in
        # statistical packages, Amer Stat, 50:361-365
        if not 0.0 <= p <= 1.0:
            raise _hatQ_8Err
        g, fj = modf(n * p + (p + 1.0)/3.0)
        j, g = min(max((int(fj), g), (1, False)), (n, False))
        q = sortedx[j - 1]
        return q + g * (sortedx[j] - q) if g else q

    try:
        return type(x)(map(_hatQ_8, probs))
    except _hatQ_8Err:
        raise ValueError("'probs' outside [0, 1]")

    
def median_iqr(seq):
    if len(seq) > 0:
        qs = quantiles(seq, (0.25, 0.5, 0.75))
        return qs[1], qs[2] - qs[0]
    else:
        return None, None
