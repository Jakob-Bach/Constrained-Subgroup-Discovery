"""Runtime competitors

Wrappers for subgroup-discovery methods in the runtime-competitor experiments.
"""


from abc import abstractmethod, ABCMeta
import time
from typing import Any, Type

import pandas as pd

import csd


class SDMethod(metaclass=ABCMeta):

    @abstractmethod
    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: int) -> Any:
        raise NotImplementedError('Abstract method.')

    def evaluate_runtime(self, X: pd.DataFrame, y: pd.Series, k: int) -> float:
        start_time = time.process_time()
        self.discover_subgroup(X=X, y=y, k=k)  # potential results ignored
        end_time = time.process_time()
        return end_time - start_time


class CSDMethod(SDMethod):

    def __init__(self, model_type: Type[csd.SubgroupDiscoverer]):
        super().__init__()
        self._model_type = model_type  # all classes support k in initializer and have fit() method

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: int) -> Any:
        model = self._model_type(k=k)
        model.fit(X=X, y=y)


class CSD_BeamSearch(CSDMethod):

    def __init__(self):
        super().__init__(model_type=csd.BeamSearchSubgroupDiscoverer)


class CSD_SMT(CSDMethod):

    def __init__(self):
        super().__init__(model_type=csd.SMTSubgroupDiscoverer)
