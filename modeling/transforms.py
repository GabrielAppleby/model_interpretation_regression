from typing import Tuple, Dict

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from modeling.none_transform import NoneTransformer

STANDARD_SCALER_NAME = "SSCALE"
KBINS_NAME = "KBINS"
NONE_TRAN_NAME = "NONE"
POWER_TRAN_NAME = "PTRAN"

STANDARD_SCALER_PARAMS: Dict = {}

KBINS_DISCRETIZER_PARAMS: Dict = {"columntransformer__kbinsdiscretizer__n_bins": [2, 5, 10, 15],
                                  "columntransformer__kbinsdiscretizer__strategy": ["quantile",
                                                                                    "uniform",
                                                                                    "kmeans"]}

POWER_TRANSFORMER_PARAMS: Dict = {}

NONE_TRANSFORMER_PARAMS: Dict = {}

TRANSFORMS: Dict[str, Tuple] = {
    STANDARD_SCALER_NAME: (StandardScaler(), STANDARD_SCALER_PARAMS, STANDARD_SCALER_NAME),
    KBINS_NAME: (KBinsDiscretizer(), KBINS_DISCRETIZER_PARAMS, KBINS_NAME),
    NONE_TRAN_NAME: (NoneTransformer(), NONE_TRANSFORMER_PARAMS, NONE_TRAN_NAME)}
