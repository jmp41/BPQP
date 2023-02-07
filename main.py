import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import time
import pickle
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_gats import GATModel
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.model.pytorch_transformer import Transformer
from train import train_epoch, test_epoch, inference
from solver import NNSolver, BPQP, QPTH, PDIPM
from model import MLP,ECONet
from utils import seed_all, dict_report, write_log
from metrics import metric_fn, e2e_loss, mse_loss, regret_loss, huber_loss
import itertools
from qpth.qp import QPFunction
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def create_loaders(args):

    with open(args['train_data'], "rb") as fh:
        df_train = pickle.load(fh)
    with open(args['valid_data'], "rb") as fh:
        df_valid = pickle.load(fh)
    with open(args['test_data'], "rb") as fh:
        df_test = pickle.load(fh)


    train_loader = DataLoader(df_train["feature"], df_train["label"], suffix = args['market']+'Train', shuffle=args['shuffle'], device = device)
    valid_loader = DataLoader(df_valid["feature"], df_valid["label"],suffix = args['market']+'Valid',shuffle=args['shuffle'], device = device)
    test_loader = DataLoader(df_test["feature"], df_test["label"],suffix = args['market']+'Test',shuffle=args['shuffle'],  device = device)

    return train_loader, valid_loader, test_loader

def create_preditor(args):
    if args['predictor'] == 'mlp':
        predictor = MLP(d_feat = args['d_feat'], hidden_size = args['hidden_size'], num_layers=args['num_layers'])
    elif args['predictor'] == 'alstm':
        predictor = ALSTMModel(args['d_feat'], args['hidden_size'], args['num_layers'], args['dropout'], 'LSTM')
    elif args['predictor'] == 'transformer':
        predictor = Transformer(args['d_feat'], args['hidden_size'], args['num_layers'], dropout=0.5)
    else:
        predictor = ECONet()
    return predictor.to(device)

def create_solver(args):
    if args['solver'] == 'naive_nn':
        args['grad_step'] = False
        solver = NNSolver(args)
        solver = solver.to(device)
    elif args['solver'] == 'dc3':
        args['grad_step'] = True
        solver= NNSolver(args)
        solver = solver.to(device)
    elif args['solver'] == 'qpth':
        solver = QPTH
    elif args['solver'] == 'primal_dual':
        solver = PDIPM(args)
    else:
        solver = BPQP(args,sign = args['sign'])
    return solver

def train_net(train_loader,valid_loader,test_loader, metric_fn,  output_path, args):
    for times in range(args['repeat']):
        write_log('create preditor...')
        model = create_preditor(args).to(device)
        solver = create_solver(args) #

        if args['solver'] == 'naive_nn':
            optimizer = optim.Adam(itertools.chain(model.parameters(),solver.parameters()), lr=args['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=args['lr'])

        stats = {}
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())

        for epoch in range(args['n_epochs']):
            write_log('Running', times, 'Epoch:', epoch)

            write_log('training...'+str(epoch) +'/'+str(args['n_epochs']))
            train_epoch(model, solver, optimizer, train_loader, args)
            torch.save(model.state_dict(), output_path + '/model.bin.e' + str(epoch))
            torch.save(optimizer.state_dict(), output_path + '/optimizer.bin.e' + str(epoch))

            write_log('evaluating...'+str(epoch) +'/'+str(args['n_epochs']))
            train_loss, train_regret, train_mse, train_score, train_precision, train_recall, train_ic, train_rank_ic, train_avg_ret, train_avg_std, train_cum_ret,train_mdd,train_icir, train_rankicir = test_epoch(model, solver,
                                                                                                    metric_fn,
                                                                                                   train_loader, args,
                                                                                                   prefix='Train')
            valid_loss,valid_regret,valid_mse, valid_score, valid_precision, valid_recall, valid_ic, valid_rank_ic, valid_avg_ret, valid_avg_std, valid_cum_ret,valid_mdd,valid_icir, valid_rankicir = test_epoch(model, solver,
                                                                                                    metric_fn,
                                                                                                   valid_loader, args,
                                                                                                   prefix='Valid')
            test_loss,test_regret,test_mse, test_score, test_precision, test_recall, test_ic, test_rank_ic, test_avg_ret, test_avg_std, test_cum_ret, test_mdd,test_icir, test_rankicir = test_epoch(model, solver,
                                                                                                    metric_fn,
                                                                                                   test_loader, args,
                                                                                                   prefix='Test')
            write_log('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, valid_loss, test_loss))
            write_log('train_regret %.6f, valid_regret %.6f, test_regret %.6f' % (train_regret, valid_regret, test_regret))
            write_log('train_mse %.6f, valid_mse %.6f, test_mse %.6f' % (train_mse, valid_mse, test_mse))
            write_log('train_score %.6f, valid_score %.6f, test_score %.6f' % (train_score, valid_score, test_score))
            write_log('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, valid_ic, test_ic))
            write_log('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (train_rank_ic, valid_rank_ic, test_rank_ic))
            write_log('train_cum_ret %.6f, valid_cum_ret %.6f, test_cum_ret %.6f' % (
            train_cum_ret, valid_cum_ret, test_cum_ret))
            write_log('train_sharpe %.6f, valid_sharpe %.6f, test_sharpe %.6f' % (
                train_avg_ret/train_avg_std*np.sqrt(252), valid_avg_ret/valid_avg_std*np.sqrt(252), test_avg_ret/test_avg_std*np.sqrt(252)))
            write_log('Train Precision: ', train_precision)
            write_log('Valid Precision: ',valid_precision)
            write_log('Test Precision: ', test_precision)
            write_log('Train Recall: ', train_recall)
            write_log('Valid Recall: ',valid_recall)
            write_log('Test Recall: ', test_recall)

            for name in ['train','valid','test']:

                dict_report(stats,name + '_loss', eval(name + '_loss'))
                dict_report(stats, name + '_regret', eval(name + '_regret'))
                dict_report(stats, name + '_mse', eval(name + '_mse'))
                dict_report(stats, name + '_score', eval(name + '_score'))
                dict_report(stats, name + '_precision', eval(name + '_precision'))
                dict_report(stats, name + '_recall', eval(name + '_recall'))
                dict_report(stats, name + '_ic', eval(name + '_ic'))
                dict_report(stats, name + '_rank_ic', eval(name + '_rank_ic'))
                dict_report(stats, name + '_rank_icir', eval(name + '_rankicir'))
                dict_report(stats, name + '_icir', eval(name + '_icir'))
                dict_report(stats, name + '_avg_ret', eval(name + '_avg_ret'))
                dict_report(stats, name + '_avg_std', eval(name + '_avg_std'))
                dict_report(stats, name + '_cum_ret', eval(name + '_cum_ret'))
                dict_report(stats, name + '_mdd', eval(name + '_mdd'))

            if valid_score > best_score:
                best_score = valid_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(model.state_dict())
            else:
                stop_round += 1
                if stop_round >= args['early_stop']:
                    write_log('early stop')
                    break
        with open(os.path.join(output_path, str(times)+'output.dict'), 'wb') as f:
            pickle.dump(stats, f)

        write_log('best score:', best_score, '@', best_epoch)
        torch.save(best_param, output_path + '/model.bin')
        model.load_state_dict(best_param)

        infe_stats = {}
        write_log('inference...'+str(times) +'/'+str(args['repeat']))

        pred = inference(model, solver, test_loader, args)
        pred.to_pickle(output_path + '/pred.pkl.' + 'test' + str(times))
        precision, recall, ic, rank_ic,avg_ret, avg_std, cum_ret, mdd, icir, rankicir = metric_fn(pred)

        write_log(('%s: IC %.6f Rank IC %.6f ICIR %.6f') % (
            'Test', ic, rank_ic, icir))
        write_log('Test', ': Precision ', precision)
        write_log('Test', ': Recall ', recall)
        write_log('Test', ': Sharpe ', avg_ret/avg_std*np.sqrt(252))
        write_log('Test', ': Cum_Ret ', cum_ret)
        write_log('Test', ': Max_Drawdown ', mdd)
        dict_report(infe_stats, 'inference_IC', ic)
        dict_report(infe_stats, 'inference_RankIC', rank_ic)
        dict_report(infe_stats, 'inference_ICIR', icir)
        dict_report(infe_stats, 'inference_RnakICIR', rankicir)
        dict_report(infe_stats, 'inference_Cum_Ret', cum_ret)
        dict_report(infe_stats, 'inference_Max_Drawdown', mdd)
        dict_report(infe_stats, 'inference_Sharpe', avg_ret/avg_std*np.sqrt(252)) # Recommend Qlib backtest

        write_log('save info...'+str(times) +'/'+str(args['repeat']))
        with open(os.path.join(output_path, str(times)+'infe.dict'), 'wb') as f:
            pickle.dump(infe_stats, f)

    write_log('finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BPQP')
    # model
    parser.add_argument('--predictor', type=str, default='mlp',
                        choices=['mlp', 'alstm', 'transformer'])
    parser.add_argument('--solver', type=str, default='bpqp',
                        choices=['qpth', 'dc3', 'naive_nn','bpqp', 'primal_dual'])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--sign', type=float, default=-1)
    parser.add_argument('--d_feat', type=int, default=153)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hiddenSize', type=int, default=512) # naive NN hidden layer

    # train
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fre_d', type=int, default=10)
    parser.add_argument('--loss', default='mse', choices=['e2e', 'regret', 'mse', 'huber_loss', 'softloss'])
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--zeta', type=int, default=4)
    parser.add_argument('--normed_mse', type=bool, default=True)

    # data
    parser.add_argument('--market', type=str, default='CN', choices=['US','CN'])
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--train_data', default='/home/jianming/PONet/e2epo/dataset/CN_feature_dataset_market_csi500_train_start2008-01-01_end2014-12-31')
    parser.add_argument('--valid_data', default='/home/jianming/PONet/e2epo/dataset/CN_feature_dataset_market_csi500_valid_start2015-01-01_end2016-12-31')
    parser.add_argument('--test_data', default='/home/jianming/PONet/e2epo/dataset/CN_feature_dataset_market_csi500_test_start2017-01-01_end2020-08-01')

    # other
    parser.add_argument('--seed', type=int, default=0)

    # DC3
    parser.add_argument('--max_stock', type=int, default=530)
    parser.add_argument('--grad_step', type=bool, default=True)
    parser.add_argument('--corrEps', type=float, default=1e-4)
    parser.add_argument('--corrTestMaxSteps', type=int, default=10)
    parser.add_argument('--corrMomentum', type=float, default=0)
    parser.add_argument('--softWeightEqFrac', type=float, default=0.5)

    args = parser.parse_args()
    args = vars(args)  # change to dictionary
    print(args)

    seed_all(args['seed'])

    save_dir = os.path.join('results', str(args['market']), 'method', str(args['predictor']),str(args['solver']),str(args['loss']),
                            str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Load data
    print('load data')
    train_loader, valid_loader, test_loader = create_loaders(args)

    # Run method
    print('run training')

    train_net(train_loader, valid_loader, test_loader, metric_fn, save_dir, args)
