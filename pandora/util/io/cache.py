from collections import OrderedDict
from inspect import signature
from json import JSONEncoder
from pathlib import Path
import pandas as pd
import numpy as np
import functools
import operator
import hashlib
import pickle
import json

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


def _hash(obj):
    return hashlib.md5(obj).hexdigest()


class Cache:
    def __init__(self, dir_path: str, rerun: bool = False, with_param: bool = False):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)
        self.with_param = with_param
        self.rerun = rerun

    def __call__(self, func):
        func_name = func.__name__

        def wrapper(*args, **kwargs):
            sig = signature(func)
            # ignore default value
            bound_args = sig.bind(*args, **kwargs)
            unique_id: str = self._get_unique_id(bound_args.arguments)
            path: Path = self.dir_path.joinpath(f'{func_name}_{unique_id}')

            ret = Cache._read_cache(path, rerun=self.rerun)
            if ret is None:
                ret = func(*args, **kwargs)
                Cache._write(path, ret)
            return ret

        return wrapper

    @staticmethod
    def _write(path, obj):
        # TODO: FileProcessor
        if isinstance(obj, pd.DataFrame):
            path = f'{path}.feather'
            obj.to_feather(str(path))
        else:
            path = f'{path}.pickle'
            with open(str(path), 'wb') as f:
                pickle.dump(obj, f, protocol=4)

    @staticmethod
    def _read_cache(path: Path, rerun):
        if rerun:
            return None
        if Path(f'{path}.pickle').exists():
            return pickle.load(open(f'{path}.pickle', 'rb'))
        if Path(f'{path}.feather').exists():
            return pd.read_feather(f'{path}.feather')
        return None

    @classmethod
    def _get_unique_id(cls, params) -> str:
        if not params:
            return 'with_no_param'
        dependencies = [f'{key}_{cls._get_hash(param)}' for key, param in sorted(params.items(), key=lambda item: str(item[0]))]
        return hashlib.md5(str(dependencies).encode()).hexdigest()

    @classmethod
    def _get_hash(cls, obj):
        if isinstance(obj, (str, int, float)):
            return cls._literals(obj)
        elif isinstance(obj, pd.DataFrame):
            return cls._data_frame(obj)
        elif isinstance(obj, np.ndarray):
            return cls._ndarray(obj)
        elif isinstance(obj, (list, dict, tuple)):
            return cls._mutables(obj)
        else:
            return _hash(pickle.dumps(obj))

    @staticmethod
    def _data_frame(obj: pd.DataFrame):
        string = str(obj.columns.tolist()) + str(obj.index) + str(obj.shape)
        return _hash(string.encode())

    @staticmethod
    def _ndarray(obj: np.ndarray):
        # not implemented
        return _hash(bytes(obj))

    @staticmethod
    def _mutables(obj):
        return _hash(json.dumps(obj, cls=_DictParamEncoder).encode())

    @staticmethod
    def _literals(obj):
        return _hash(str(obj).encode())


class _DictParamEncoder(JSONEncoder):
    """
    JSON encoder for :py:class:`~DictParameter`, which makes :py:class:`~FrozenOrderedDict` JSON serializable.
    """
    def default(self, obj):
        if isinstance(obj, FrozenOrderedDict):
            return obj.get_wrapped()
        return json.JSONEncoder.default(self, obj)


class FrozenOrderedDict(Mapping):
    """
    It is an immutable wrapper around ordered dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability and ordering are desired.
    """
    def __init__(self, *args, **kwargs):
        self.__dict = OrderedDict(*args, **kwargs)
        self.__hash = None

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        # We should use short representation for beautiful console output
        return repr(dict(self.__dict))

    def __hash__(self):
        if self.__hash is None:
            hashes = map(hash, self.items())
            self.__hash = functools.reduce(operator.xor, hashes, 0)

        return self.__hash

    def get_wrapped(self):
        return self.__dict


def recursively_freeze(value):
    """
    Recursively walks ``Mapping``s and ``list``s and converts them to ``FrozenOrderedDict`` and ``tuples``, respectively.
    """
    if isinstance(value, Mapping):
        return FrozenOrderedDict(((k, recursively_freeze(v)) for k, v in value.items()))
    elif isinstance(value, list) or isinstance(value, tuple):
        return tuple(recursively_freeze(v) for v in value)
    return value
