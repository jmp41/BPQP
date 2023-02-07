import qlib
import pickle
from qlib.config import REG_US, REG_CN
from qlib.utils import init_instance_by_config, exists_qlib_data
from qlib.model.riskmodel import StructuredCovEstimator
from qlib.data import D
import numpy as np
import pandas as pd
import os
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
provider_uri = "~/.qlib/qlib_data/cn_data"
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
# if not exists_qlib_data(provider_uri):
#     from qlib.tests.data import GetData
#     GetData().qlib_data(target_dir=provider_uri, region=REG_US)
qlib.init(provider_uri=provider_uri, region=REG_CN)

country = 'CN'
market = "csi500" # sp500
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "learn_processors": [
        {
            "class" : "DropCol",
            "kwargs":{"col_list": ["VWAP0","KUP","KUP2","HIGH0","IMIN5"]}
        },
        {
            'class': 'DropnaLabel'
        },
        {
            "class": "CSZScoreNorm",
            "kwargs": {"fields_group": "feature", "method": "robust"}
        },
        {
            "class": "CSZFillna",
            "kwargs": {"fields_group": "feature"}
        },
        {
            "class": "CSZScoreNorm",
            "kwargs": {"fields_group": "label", "method": "robust"}
        }
    ],
    "infer_processors": [
        {
            "class" : "DropCol",
            "kwargs":{"col_list": ["VWAP0","KUP","KUP2","HIGH0","IMIN5"]}
        },
        {
            "class": "RobustZScoreNorm",
            "kwargs": {"fields_group": "feature", "clip_outlier": True}
        },
        {
            "class": "CSZFillna",
            "kwargs": {"fields_group": "feature"}
        }
    ],
    'label': [
        'Ref($close, -1) / $close - 1'
    ]
}

dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01")
            }
        }
}

def get_daily_code(df):
    return df.reset_index(level=0).index.values

def roubst_z_score(df):
    return (df - df.mean())/(0.0001+df.std())

def prepare_risk_data(df_index, suffix = 'Train',T=240, start_time = "2007-01-01", riskdata_root = './riskdata'):
    riskmodel = StructuredCovEstimator()
    codes = df_index.groupby("datetime").apply(get_daily_code)
    ret_date = codes.index.values

    price_all = (
        D.features(D.instruments("all"), ["$close"], start_time=start_time, end_time=ret_date[-1]).squeeze().unstack(level="instrument")
    )
    cur_idx = np.argwhere(price_all.index==ret_date[0])[0][0]
    assert cur_idx - T + 1 >= 0
    for i in range(len(ret_date)):
        date = pd.Timestamp(ret_date[i])
        print(date)
        ref_date = price_all.index[i+cur_idx - T + 1]
        code = codes[i]

        price = price_all.loc[ref_date:date, code]

        ret = price.pct_change()
        ret = ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)
        if suffix=='CNTrain':
            ret.groupby("datetime").apply(roubst_z_score)

        cov_estimated = riskmodel.predict(ret, is_price=False, return_decomposed_components=False)

        root = riskdata_root +suffix+ "/" + date.strftime("%Y%m%d")
        os.makedirs(root, exist_ok=True)
        cov_estimated.to_pickle(root + "/factor_exp.pkl")

dataset = init_instance_by_config(dataset_config)
df_train = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L).fillna(method = 'ffill')
df_valid = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I).fillna(method = 'ffill')
df_test = dataset.prepare(
        "test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I).fillna(method = 'ffill')

with open("./{}_feature_dataset_market_{}_{}_start{}_end{}".format(country,market, 'train', "2008-01-01", "2014-12-31"), 'wb') as f:
    pickle.dump(df_train, f)
with open("./{}_feature_dataset_market_{}_{}_start{}_end{}".format(country,market,'valid', "2015-01-01", "2016-12-31"), 'wb') as f:
    pickle.dump(df_valid, f)
with open("./{}_feature_dataset_market_{}_{}_start{}_end{}".format(country,market, 'test', "2017-01-01", "2020-08-01"), 'wb') as f:
    pickle.dump(df_test, f)

print("Preparing features done!")

prepare_risk_data(df_train, suffix = 'CNTrain',T=240, start_time = "2007-01-01")
prepare_risk_data(df_valid, suffix = 'CNValid',T=240, start_time = "2014-01-01")
prepare_risk_data(df_test, suffix = 'CNTest',T=240, start_time = "2015-01-01")
print("preparing risk data done!")