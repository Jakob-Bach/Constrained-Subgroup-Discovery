"""Runtime competitors

Wrappers for subgroup-discovery methods in the runtime-competitor experiments.
"""


from abc import abstractmethod, ABCMeta
import time
from typing import Any, Optional, Type
import warnings

import pandas as pd
import pysubdisc
import pysubgroup
import sd4py
import subgroups

import csd


BEAM_WIDTH = 10  # for beam search; consistent with our own default beam width
NUM_BINS = 50  # for methods that require discretization of features


def discretize_data(X: pd.DataFrame, num_bins: int = NUM_BINS) -> pd.DataFrame:
    # Equal-frequency binning, integer labels (turned into strings)
    return X.apply(pd.qcut, axis='columns', q=num_bins, labels=False, duplicates='drop').astype(str)


class SDMethod(metaclass=ABCMeta):

    @abstractmethod
    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        raise NotImplementedError('Abstract method.')

    def evaluate_runtime(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> float:
        start_time = time.process_time()
        self.discover_subgroup(X=X, y=y, k=k)  # potential results ignored
        end_time = time.process_time()
        return end_time - start_time


class CSDMethod(SDMethod):

    def __init__(self, model_type: Type[csd.SubgroupDiscoverer]):
        super().__init__()
        self._model_type = model_type  # all classes support k in initializer and have fit() method

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        model = self._model_type(k=k)
        model.fit(X=X, y=y)


class CSD_BeamSearch(CSDMethod):

    def __init__(self):
        super().__init__(model_type=csd.BeamSearchSubgroupDiscoverer)


class CSD_SMT(CSDMethod):

    def __init__(self):
        super().__init__(model_type=csd.SMTSubgroupDiscoverer)


class PysubdiscMethod(SDMethod):

    def __init__(self, search_strategy: str):
        super().__init__()
        self._search_strategy = search_strategy

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        if k is None:
            k = X.shape[1]
        data = pd.concat((X, y.astype(str)), axis='columns')
        model = pysubdisc.singleNominalTarget(data, targetColumn='target', targetValue='1')
        # Model parameters explained in https://github.com/SubDisc/SubDisc/wiki and
        # https://github.com/SubDisc/SubDisc/blob/main/manual/manual.pdf
        model.minimumCoverage = 1  # lower bound for subgroup size (absolute)
        model.maximumCoverageFraction = 1.0  # upper bound for subgroup size (relative)
        model.maximumSubgroups = BEAM_WIDTH  # according to manual, also for intermediate solutions
        model.nrThreads = 1
        model.numericStrategy = 'NUMERIC_BEST'  # enumerate all possible split points
        model.qualityMeasure = 'WRACC'
        model.qualityMeasureMinimum = -float('inf')
        model.searchDepth = k  # selected features <= search depth
        model.searchStrategy = self._search_strategy
        model.searchStrategyWidth = BEAM_WIDTH  # difference to "maximumSubgroups" unclear
        model.run(verbose=False)


class Pysubdisc_Beam(PysubdiscMethod):

    def __init__(self):
        super().__init__(search_strategy='BEAM')


class Pysubdisc_BestFirst(PysubdiscMethod):

    def __init__(self):
        super().__init__(search_strategy='BEST_FIRST')


class Pysubdisc_BreadthFirst(PysubdiscMethod):

    def __init__(self):
        super().__init__(search_strategy='BREADTH_FIRST')


class Pysubdisc_DepthFirst(PysubdiscMethod):

    def __init__(self):
        super().__init__(search_strategy='DEPTH_FIRST')


class PysubgroupMethod(SDMethod):

    def __init__(self, model: Any):
        super().__init__()
        self._model = model  # don't have a common superclass, but all have execute() method

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        if k is None:
            k = X.shape[1]
        data = pd.concat((X, y), axis='columns')
        target = pysubgroup.BinaryTarget(target_attribute='target', target_value=1)
        search_space = []
        for feature in X.columns:
            for value in X[feature].unique():
                search_space.append(pysubgroup.IntervalSelector(feature, value, float('inf')))
                search_space.append(pysubgroup.IntervalSelector(feature, float('-inf'), value))
        task = pysubgroup.SubgroupDiscoveryTask(
            data=data, target=target, search_space=search_space, result_set_size=1,
            qf=pysubgroup.WRAccQF(), depth=k  # selected features <= search depth
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="invalid value encountered in divide")
            self._model.execute(task=task)


class Pysubgroup_Apriori(PysubgroupMethod):

    def __init__(self):
        super().__init__(model=pysubgroup.Apriori())


class Pysubgroup_BeamSearch(PysubgroupMethod):

    def __init__(self):
        super().__init__(model=pysubgroup.BeamSearch(beam_width=BEAM_WIDTH))


class Pysubgroup_GpGrowth(PysubgroupMethod):

    def __init__(self):
        super().__init__(model=pysubgroup.GpGrowth())


class Pysubgroup_SimpleDFS(PysubgroupMethod):

    def __init__(self):
        super().__init__(model=pysubgroup.SimpleDFS())


class SD4PyMethod(SDMethod):

    def __init__(self, method: str):
        super().__init__()
        self._method = method

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        if k is None:
            k = X.shape[1]
        data = pd.concat((discretize_data(X=X), y), axis='columns')
        # WRAcc as "qf" doesn't work (ClassCastException), but "ps" is propoptional; binning is
        # mandatory; "k=10" is number of solutions; "max_selectors" counts number of features
        sd4py.discover_subgroups(
            ontology=data, target='target', target_value=1, qf='ps', minqual=-float('inf'),
            method=self._method, nbins=NUM_BINS, k=BEAM_WIDTH, max_selectors=k
        )


class SD4Py_Beam(SD4PyMethod):

    def __init__(self):
        super().__init__(method='beam')


class SD4Py_BSD(SD4PyMethod):

    def __init__(self):
        super().__init__(method='bsd')


class SD4Py_SDMap(SD4PyMethod):

    def __init__(self):
        super().__init__(method='sdmap')


class Subgroups_BSD(SDMethod):

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        if k is None:
            k = X.shape[1]
        data = pd.concat((discretize_data(X=X), y.astype(str)), axis='columns')
        model = subgroups.algorithms.BSD(
            min_support=0, quality_measure=subgroups.quality_measures.WRAcc(),
            optimistic_estimate=subgroups.quality_measures.WRAccOptimisticEstimate1(),
            num_subgroups=BEAM_WIDTH, max_depth=k
        )
        model.fit(pandas_dataframe=data, tuple_target_attribute_value=('target', '1'))


class Subgroups_SDMap(SDMethod):

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        data = pd.concat((discretize_data(X=X), y.astype(str)), axis='columns')
        model = subgroups.algorithms.SDMap(
            quality_measure=subgroups.quality_measures.WRAcc(),
            minimum_quality_measure_value=-float('inf'), minimum_n=0
        )  # feature-cardinality threshold not supported
        model.fit(pandas_dataframe=data, target=('target', '1'))


class Subgroups_SDMapStar(SDMethod):

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        data = pd.concat((discretize_data(X=X), y.astype(str)), axis='columns')
        model = subgroups.algorithms.SDMapStar(
            quality_measure=subgroups.quality_measures.WRAcc(),
            optimistic_estimate=subgroups.quality_measures.WRAccOptimisticEstimate1(),
            minimum_quality_measure_value=-float('inf'), minimum_n=0
        )  # feature-cardinality threshold not supported
        model.fit(pandas_dataframe=data, target=('target', '1'))


class Subgroups_VLSD(SDMethod):

    def discover_subgroup(self, X: pd.DataFrame, y: pd.Series, k: Optional[int]) -> Any:
        data = pd.concat((discretize_data(X=X), y.astype(str)), axis='columns')
        model = subgroups.algorithms.VLSD(
            quality_measure=subgroups.quality_measures.WRAcc(),
            optimistic_estimate=subgroups.quality_measures.WRAccOptimisticEstimate1(),
            q_minimum_threshold=-float('inf'), oe_minimum_threshold=-float('inf')
        )  # feature-cardinality threshold not supported
        model.fit(pandas_dataframe=data, target=('target', '1'))
