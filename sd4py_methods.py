"""SD4Py subgroup-discovery methods

Classes wrapping subgroup-discovery methods from the external package "SD4Py" to integrate them
into our main experimental pipeline. Note that these methods have a mandatory discretization step
for numeric features, so their search space differs from that of our package "csd".

Literature
----------
Hudson (2023): "Subgroup Discovery with SD4PY"
"""


import time
from typing import Any, Dict, Optional

import pandas as pd
import sd4py

import csd


NUM_BIN_LIST = [2, 3, 4, 5, 10, 15, 20, 30, 40, 50]  # for discretization


class SD4PySubgroupDiscoverer(csd.SubgroupDiscoverer):
    """Subgroup-discovery method from SD4Py

    Wrapper for methods from the package "SD4Py". Since the latter always discretizes numeric
    features (in particular, it applies equal-width binning), we iterate over a few bin numbers
    and choose the one with the highest subgroup quality.
    """

    def __init__(self, method: str, k: Optional[int] = None):
        super().__init__()
        self._method = method
        self._k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        data = pd.concat((X, y), axis='columns')  # SD4Py expects one data frame as input
        k = X.shape[1] if self._k is None else self._k  # if "k" not set, set to number of features
        start_time = time.process_time()  # as in "csd", measure time without pre-/postprocessing
        opt_quality = float('-inf')
        opt_selectors = None  # selectors are the conditions in the subgroup description
        for num_bins in NUM_BIN_LIST:
            # "ps" as "qf" is propoptional to WRAcc (which can also be set but throws exception);
            # https://sourceforge.net/p/vikamine/code/HEAD/tree/trunk/org.vikamine.kernel/src/org/vikamine/kernel/subgroup/quality/functions/PiatetskyShapiroQF.java
            # "k=1" is number of solutions, while "max_selectors" is number of features!
            sd_results = sd4py.discover_subgroups(
                    ontology=data, target='target', target_value=1, qf='ps', minqual=float('-inf'),
                    method=self._method, nbins=num_bins, k=1, max_selectors=k
            )
            cand_quality = sd_results.subgroups[0].quality  # "subgroups" is list of length 1
            if cand_quality > opt_quality:
                opt_quality = cand_quality
                opt_selectors = sd_results.subgroups[0].selectors
        end_time = time.process_time()
        self._box_lbs = pd.Series([float('-inf')] * X.shape[1], index=X.columns)
        self._box_ubs = pd.Series([float('inf')] * X.shape[1], index=X.columns)
        for selector in opt_selectors:  # each feature only occurs in zero or one selector
            self._box_lbs[selector.attribute] = selector.lower_bound
            self._box_ubs[selector.attribute] = selector.upper_bound
        return {'objective_value': opt_quality,
                'optimization_status': None,
                'optimization_time': end_time - start_time}

    # Prediction overridden since UBs are exclusive in SD4Py ("csd": both bounds inclusive)
    # https://sourceforge.net/p/vikamine/code/HEAD/tree/trunk/org.vikamine.kernel/src/org/vikamine/kernel/subgroup/selectors/SGSelectorGenerator.java#l181
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series((X.ge(self.get_box_lbs()) & X.lt(self.get_box_ubs())).all(
            axis='columns').astype(int), index=X.index)


class BSDSubgroupDiscoverer(SD4PySubgroupDiscoverer):
    """BSD for subgroup discovery

    An exhaustive search method.

    Literature
    ----------
    Lemmerich et al. (2010): "Fast Discovery of Relevant Subgroup Patterns"
    """

    def __init__(self, k: Optional[int] = None):
        super().__init__(method='bsd', k=k)


class SDMapSubgroupDiscoverer(SD4PySubgroupDiscoverer):
    """SD-Map for subgroup discovery

    An exhaustive search method.

    Literature
    ----------
    Atzmueller & Puppe (2006): "SD-Map - A Fast Algorithm for Exhaustive Subgroup Discovery"
    """

    def __init__(self, k: Optional[int] = None):
        super().__init__(method='sdmap', k=k)
