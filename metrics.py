# Backtest
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import WeightStrategyBase
from qlib.contrib.strategy import TopkDropoutStrategy
import pandas as pd
import numpy as np
import torch

def metric_fn(preds):
    preds.index.name = 'datetime'
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby('datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby('datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    ic = preds.groupby('datetime').apply(lambda x: x.label.corr(x.score)).mean()
    icir = ic/preds.groupby('datetime').apply(lambda x: x.label.corr(x.score)).std()
    rank_ic = preds.groupby('datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()
    rank_icir = rank_ic/preds.groupby('datetime').apply(lambda x: x.label.corr(x.score,method='spearman')).std()
    try:
        avg_ret = preds.groupby('datetime').apply(lambda x: x.label.dot(x.weight_pred)).mean()
        avg_std = preds.groupby('datetime').apply(lambda x: x.label.dot(x.weight_pred)).std()
        cum_ret = (preds.groupby('datetime').apply(lambda x: x.label.dot(x.weight_pred))+1).cumprod()[-1] - 1
        ret = preds.groupby('datetime').apply(lambda x: x.label.dot(x.weight_pred))
        mdd = MaxDrawdown(ret)
    except:
        avg_ret,avg_std,cum_ret,mdd = np.nan,np.nan,np.nan,np.nan
    return precision, recall, ic, rank_ic, avg_ret, avg_std, cum_ret,mdd,icir,rank_icir

def obj_fn(weight,rets, variance, args):
    return weight.T@rets - 0.5*args['sigma']*weight.T@variance@weight

def regret_loss(weight_pred, exact_weight, pred, y, variance, args):
    return (obj_fn(weight_pred, pred, variance,args) - obj_fn(exact_weight,y, variance,args))**2

def mse_loss(pred,y, args):
    return torch.mean((pred - y)**2)

def huber_loss(weight_pred, pred, y, variance, args):
    reg_l = (1/args['gamma'])*mse_loss(pred,y,args) - obj_fn(weight_pred,y,variance,args)
    if reg_l > args['zeta']**2:
        return args['zeta']*(reg_l - args['zeta'])
    else:
        return reg_l

def e2e_loss(weight_pred, exact_weight, pred, y, variance, args):
    gamma = args['gamma']
    assert gamma>0
    return regret_loss(weight_pred, exact_weight, pred, y, variance, args) + mse_loss(pred, y, args) * (1 / gamma)

def soft_loss(weight_pred,pred, y,variance,args):
    return mse_loss(pred,y,args)-(weight_pred.T@y)

def MaxDrawdown(ret):
    return_list = np.cumprod(1+ret)
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # start
    return (return_list[j] - return_list[i]) / (return_list[j])
