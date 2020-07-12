from typing import Callable
from collections import defaultdict


class Beholder(type):
    __inheritors__ = defaultdict(dict)

    def __new__(mcs, name, bases, dct):
        klass = type.__new__(mcs, name, bases, dct)

        for base in klass.mro()[1:-1]:
            mcs.__inheritors__[base][Beholder._process(klass.__name__)] = klass

        return klass

    @staticmethod
    def _process(name):
        return name.lower().replace('-', '_')

    @property
    def __modules__(cls):
        return cls.__inheritors__[cls]

    def get(cls, name, default=None, process: Callable = None):
        return cls.__modules__.get((process or cls._process)(name), default)
