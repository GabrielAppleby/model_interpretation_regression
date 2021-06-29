from typing import Dict, Tuple

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

KNN_NAME = "KNN"
SVR_NAME = "SVR"
XGB_NAME = "XGB"
LIN_NAME = "LIN"
MEAN_NAME = "MEAN"
MEDIAN_NAME = "MEDIAN"

KNN_PARAMS: Dict = {"kneighborsregressor__n_neighbors": [1, 5, 10, 15, 20, 25]}

SVR_PARAMS: Dict = {"svr__C": [.01, .5, 1, 1.5, 2, 2.5]}

XGB_PARAMS: Dict = {"xgbregressor__max_depth": [3, 5, 7, 9]}

LINEAR_PARAMS: Dict = {}

REGRESSORS: Dict[str, Tuple] = {KNN_NAME: (KNeighborsRegressor(), KNN_PARAMS, KNN_NAME),
                                SVR_NAME: (SVR(), SVR_PARAMS, SVR_NAME),
                                XGB_NAME: (XGBRegressor(), XGB_PARAMS, XGB_NAME),
                                LIN_NAME: (LinearRegression(), LINEAR_PARAMS, LIN_NAME),
                                MEAN_NAME: (DummyRegressor(strategy='mean'), {}, MEAN_NAME),
                                MEDIAN_NAME: (DummyRegressor(strategy='median'), {}, MEDIAN_NAME)}
