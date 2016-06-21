__author__ = 'Robbert'

import itertools
if __name__ == "__main__":
    l = [[(1,2,3),(3,4,5)],[(6,7,8)]]
    print list(itertools.chain.from_iterable(l))