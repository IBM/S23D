from enum import Enum
from functools import partial

BUFF_SIZE = 10240000


class Compressor(Enum):
    BZ2 = 0


def factory(opt):
    if opt is None:
        return partial(open, buffering=BUFF_SIZE)

    if opt.compressor == Compressor.BZ2:
        import bz2

        if opt.options:
            return partial(bz2.open, **opt.options)
        else:
            return bz2.open
    else:
        raise ValueError("Wrong compression alg.")
