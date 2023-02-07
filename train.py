import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from metrics import regret_loss, mse_loss, e2e_loss, soft_loss, huber_loss, MaxDrawdown
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_epoch(model, solver, optimizer, train_loader, args):

    model.train()
    running_loss = 0
    for i, slc in tqdm(train_loader.iter_daily(), total=train_loader.daily_length):
        # optimizer.step()
        feature, label, variance , stock_index, _ = train_loader.get(i, slc)

        # predict returns
        pred = model(feature)

        # differentiable solver
        # loss
        if args['loss'] == 'e2e':
            weight_pred = solver(variance, pred)
            exact_weight = solver(variance, label)
            loss = e2e_loss(weight_pred, exact_weight, pred, label, variance, args) # supervised by ground truth weight
        elif args['loss'] == 'regret':
            weight_pred = solver(variance, pred)
            exact_weight = solver(variance, label)
            loss = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
        elif args['loss'] == 'mse':
            loss = mse_loss(pred, label, args)
        elif args['loss'] == 'huber_loss':
            weight_pred = solver(variance, pred)
            loss = huber_loss(weight_pred, pred, label, variance, args)
        else:
            weight_pred = solver(variance, pred)
            loss = soft_loss(weight_pred, pred, label, variance, args)
        running_loss += loss
        if i % args['fre_d']==0 and i>0:
            running_loss = running_loss/args['fre_d']
            running_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
            optimizer.zero_grad()
            running_loss = 0

def test_epoch(model, solver, metric_fn, test_loader, args, prefix='Test'):

    model.eval()

    losses = []
    regrets = []
    mse = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, variance , stock_index, _ = test_loader.get(i, slc)

        with torch.no_grad():

            pred = model(feature)

            if args['loss'] == 'e2e':
                weight_pred = solver(variance, pred)
                exact_weight = solver(variance,label)
                loss = e2e_loss(weight_pred, exact_weight, pred, label, variance, args)  # supervised by ground truth weight
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance,
                                     args)
                _mse = mse_loss(pred, label, args)
                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),
                                           'weight_pred': weight_pred.cpu().numpy(),
                                           'exact_weight': exact_weight.cpu().numpy()},
                                          index=[test_loader.get_daily_date(i)]*len(pred)))
            elif args['loss'] == 'regret':
                weight_pred = solver(variance, pred)
                exact_weight = solver(variance, label)
                loss =regret_loss(weight_pred, exact_weight, pred, label, variance,
                                args)  # supervised by ground truth weight
                regret = loss
                _mse = mse_loss(pred, label, args)
                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),
                                           'weight_pred': weight_pred.cpu().numpy(),
                                           'exact_weight': exact_weight.cpu().numpy()},
                                          index=[test_loader.get_daily_date(i)] * len(pred)))
            elif args['loss'] == 'mse':
                if prefix=='Train':
                    weight_pred = solver(variance, pred)
                    exact_weight = solver(variance, label)

                    regret = regret_loss(weight_pred, exact_weight, pred, label, variance,
                                args)
                else:
                    regret = torch.zeros(1).to(device)
                loss = mse_loss(pred, label, args)
                _mse = loss
                # regret = torch.zeros(1).to(device) # To accelerate, do not calculate regret
                preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy()}, index=[test_loader.get_daily_date(i)]*len(pred)))
            elif args['loss']=='huber_loss':
                exact_weight = solver(variance, label)
                weight_pred = solver(variance, pred)
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance,
                                                     args) # To accelerate, do not calculate regret
                loss = huber_loss(weight_pred, pred, label,variance, args)
                _mse = mse_loss(pred, label, args)
                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),
                                           'weight_pred': weight_pred.cpu().numpy()},
                                          index=[test_loader.get_daily_date(i)] * len(pred)))
            else:
                weight_pred = solver(variance, pred)
                exact_weight = solver(variance, label)
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance,
                                     args)
                loss = soft_loss(weight_pred, pred, label,variance, args)
                _mse = mse_loss(pred, label, args)
                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),
                                           'weight_pred': weight_pred.cpu().numpy(),
                                           'exact_weight': exact_weight.cpu().numpy()},
                                          index=[test_loader.get_daily_date(i)] * len(pred)))
        regrets.append(regret.item())
        mse.append(_mse.item())
        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic, avg_ret, avg_std, cum_ret,mdd,icir, rankicir = metric_fn(preds)
    if args['loss'] == 'mse':
        scores = ic
    else:
        scores = avg_ret/avg_std * np.sqrt(252)

    return np.mean(losses),np.mean(regrets),np.mean(mse), scores, precision, recall, ic, rank_ic,avg_ret, avg_std, cum_ret,mdd,icir,rankicir

def inference(model, solver, data_loader, args):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, variance , stock_index, _ = data_loader.get(i, slc)
        with torch.no_grad():
            pred = model(feature)
            weight_pred = solver(variance, pred)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), 'weight_pred': weight_pred.cpu().numpy() }, index=[data_loader.get_daily_date(i)]*len(pred)))

    preds = pd.concat(preds, axis=0)
    return preds