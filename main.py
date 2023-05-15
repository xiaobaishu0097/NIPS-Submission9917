import argparse
import json
import math
import os
import random
import time
from argparse import ArgumentParser, Namespace
from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import GetData, GetData1
from models import (BaselineRNN, FullBasicModel, FullBasicModel1, gstarhat,
                    gstarhat1)
from utils.logger import Logger

random.seed(12)
# T = 100
# N = 100


def get_training_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--n_nodes', type=int, default=100)
    parser.add_argument('--n_timepoints', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=2000)

    parser.add_argument('--work-dir', type=str, default='./work_dirs/')
    parser.add_argument('--logger-name',
                        type=str,
                        default='NeurIPS2023Accepted')
    return parser


def estimate_adjacency_matrix(n_nodes: int) -> Tuple[np.ndarray, float]:
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    for row in range(0, n_nodes, 1):
        for coloumn in range(row + 1, n_nodes, 1):
            probtemp = np.random.multinomial(
                1,
                pvals=[
                    30 / n_nodes, 0.5 * n_nodes**(-0.7), 0.5 * n_nodes**(-0.7),
                    1 - 30 / n_nodes - n_nodes**(-0.7)
                ],
                size=1)
            if probtemp[0][0] == 1:
                adjacency_matrix[row, coloumn] = adjacency_matrix[coloumn,
                                                                  row] = 1
            elif probtemp[0][1] == 1:
                adjacency_matrix[row, coloumn] = 1
            elif probtemp[0][2] == 1:
                adjacency_matrix[coloumn, row] = 1
    column_sums = sum(adjacency_matrix.T)
    return adjacency_matrix, column_sums


def train_gstarhat_model(model, train_dataloader, criterion, optimizer,
                         epochs) -> nn.Module:
    for _ in range(epochs):
        for batch_X, batch_Y in train_dataloader:
            input_data = batch_X.cuda()
            target = batch_Y.cuda()

            prediction = model(input_data)
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train(args: Namespace) -> float:
    logger = Logger.get_logger(args.logger_name)

    adjacency_matrix, ni = estimate_adjacency_matrix(args.n_nodes)

    weights = np.dot(np.diag(1 / ni),
                     adjacency_matrix)  # W: normalized adjacency matrix

    beta = np.array([0.2, 0.3, -0.1])
    G = beta[1] * np.identity(args.n_nodes) + beta[2] * weights

    sigma = 0.5**abs(
        np.array([0, 1, 2, 3, 4] * 5).reshape((5, 5)) -
        np.array([0, 1, 2, 3, 4] * 5).reshape((5, 5)).T)
    Z = np.random.multivariate_normal(np.ones(5), sigma, args.n_nodes)
    gamma = np.array([-0.5, 0.3, 0.8, -0.1, -0.1])
    # B0 = beta[0] + Z.dot(gamma)
    def g0(z):
        return(0.2-0.5*z[:,0]+0.3*z[:,1]+0.8*z[:,2]-0.1*z[:,3]-0.1*z[:,4])                       # case 1
        #return(5-2*z[:,0]+0.5*z[:,1]**2-z[:,2]**3-np.log(z[:,3]+3)+np.sqrt(z[:,4]+3))     # case 2
        # return(z[:,0]**2-2*z[:,1]**2+z[:,1]*z[:,2]+z[:,3]*z[:,4])                       # case 3
    B0=g0(Z)

    mu0 = np.linalg.inv((1 - beta[1]) * np.identity(args.n_nodes) -
                        beta[2] * weights).dot(B0)
    sig = 1
    cov0 = sig**2 * np.linalg.inv(
        np.identity(args.n_nodes**2) - np.kron(G, G)).dot(
            np.ravel(np.identity(args.n_nodes))).reshape(
                args.n_nodes, args.n_nodes).T
    Y0 = np.random.multivariate_normal(mu0, cov0, 1)

    W_tensor = torch.from_numpy(weights).clone().detach()

    coverage_probability_count = np.zeros((2, 3))

    simu_loss = []

    for simu in range(0, 50):
        random.seed(simu)
        Y = np.zeros((args.n_nodes, args.n_timepoints * 2))
        epsilon = np.random.multivariate_normal(
            np.zeros(args.n_nodes), sig * np.identity(args.n_nodes),
            2 * args.n_timepoints)
        Y[:, 0] = Y0
        Y[:, 1] = B0 + G.dot(Y0.T)[:, 0] + epsilon[0, :]

        for i in range(0, 2 * args.n_timepoints - 2):
            Y[:, i + 2] = B0 + G.dot(Y[:, i + 1]) + epsilon[i + 1, :]
        YY = Y[:, (args.n_timepoints - 1):2 * args.n_timepoints]

        p = 5
        XX = np.zeros((args.n_timepoints, args.n_nodes, p + 3))
        CC = np.zeros((p + 3, p + 3))
        BB = np.zeros((p + 3, 1))

        MSE = 0
        for t in range(0, args.n_timepoints):
            XX[t, :, 0] = np.ones(args.n_nodes)
            XX[t, :, 1] = YY[:, t].ravel()
            XX[t, :, 2] = weights.dot(YY[:, t])
            XX[t, :, 3:8] = Z
            CC = CC + np.dot(XX[t, :, :].T, XX[t, :, :])
            BB = BB + XX[t, :, :].T * (np.mat(YY[:, t + 1]).T)
        thetahat = np.linalg.inv(CC) * BB

        for t in range(0, args.n_timepoints):
            MSE = MSE + np.sum(
                (YY[:, t + 1] - np.ones(args.n_nodes) * thetahat[0, 0] -
                 np.ravel((Z.dot(thetahat[3:8, 0]))) -
                 (thetahat[1, 0] * np.identity(args.n_nodes) +
                  thetahat[2, 0] * weights).dot(Y[:, t]))**2)
        sig2zhu = MSE / args.n_nodes / args.n_timepoints
        se2zhu = np.linalg.inv(CC) * sig2zhu
        if thetahat[0] >= beta[0] - 1.96 * np.sqrt(se2zhu[0, 0]) and thetahat[
                0] <= beta[0] + 1.96 * np.sqrt(se2zhu[0, 0]):
            coverage_probability_count[1, 0] += 1
        if thetahat[1] >= beta[1] - 1.96 * np.sqrt(se2zhu[1, 1]) and thetahat[
                1] <= beta[1] + 1.96 * np.sqrt(se2zhu[1, 1]):
            coverage_probability_count[1, 1] += 1
        if thetahat[2] >= beta[2] - 1.96 * np.sqrt(se2zhu[2, 2]) and thetahat[
                2] <= beta[2] + 1.96 * np.sqrt(se2zhu[2, 2]):
            coverage_probability_count[1, 2] += 1

        Y_tensor = torch.from_numpy(YY).clone().detach().float()
        XX_tensor = torch.from_numpy(XX).clone().detach().float()

        train_set = GetData(XX_tensor[0:(args.n_timepoints - args.n_timepoints // 10), ...],
                            Y_tensor[:, 1:(args.n_timepoints + 1 - args.n_timepoints // 10)])
        train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)

        test_set = GetData(XX_tensor[(args.n_timepoints - args.n_timepoints // 10):, ...],
                            Y_tensor[:, (args.n_timepoints + 1 - args.n_timepoints // 10):(args.n_timepoints + 1)])
        test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True)

        criterion = nn.MSELoss().cuda()

        # Here to change the model structure
        model = BaselineRNN(input_size=p, hidden_size=p*2, output_size=1, n_layers=4).cuda()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)

        def train_model(model, dataloader, criterion, optimizer, n_epochs=25):
            for _ in range(n_epochs):
                for input, target in dataloader:
                    input = input.cuda()
                    target = target.cuda()
                    output, hidden_states = model(input)

                    loss = criterion(einops.rearrange(output, 'b t 1 -> b t'),
                                     target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        def test_model(model, dataloader, criterion):
            model.eval()
            with torch.no_grad():
                loss_sum = []
                for input, target in dataloader:
                    input = input.cuda()
                    target = target.cuda()
                    output, hidden_states = model(input)

                    loss = criterion(einops.rearrange(output, 'b t 1 -> b t'),
                                     target)
                    loss_sum.append(loss.item())
            return np.mean(loss_sum)

        train_model(model, train_dataloader, criterion, optimizer, 100)
        loss = test_model(model, test_dataloader, criterion)
        logger.info(f'Simu: {simu}; MSE: {MSE / args.n_nodes / args.n_timepoints}, RNN loss: {loss.item()}')
        simu_loss.append(loss.item())
    logger.info(f'Average loss: {np.mean(simu_loss)}')

    return np.mean(simu_loss)


        # thetahat_tensor = torch.from_numpy(thetahat)

        # model = FullBasicModel1(p, 100, 100, 100, 1, W_tensor, args.n_nodes, p,
        #                         thetahat_tensor).cuda()
        # optimizer = torch.optim.Adam(model.parameters(), 0.001)

        # for i in range(args.epochs):
        #     for batch_X, batch_Y in train_dataloader:
        #         input_data = batch_X.cuda()
        #         target = batch_Y.cuda()
        #         out1, out2 = model(input_data)

        #         loss = criterion(out1, target)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        # XXX_tensor = torch.mean(XX_tensor, dim=0)
        # train_set1 = GetData1(XXX_tensor[:, 3:8], XXX_tensor[:, 1])
        # train_set2 = GetData1(XXX_tensor[:, 3:8], XXX_tensor[:, 2])
        # train_dataloader1 = DataLoader(train_set1,
        #                                batch_size=128,
        #                                shuffle=True)
        # train_dataloader2 = DataLoader(train_set2,
        #                                batch_size=128,
        #                                shuffle=True)

        # model1 = gstarhat1(p, 100, 100, 100, 1, W_tensor, args.n_nodes, p,
        #                    thetahat_tensor).cuda()

        # model2 = gstarhat1(p, 100, 100, 100, 1, W_tensor, args.n_nodes, p,
        #                    thetahat_tensor).cuda()

        # optimizer1 = torch.optim.Adam(model1.parameters(), 0.001)
        # optimizer2 = torch.optim.Adam(model2.parameters(), 0.001)

        # model1 = train_gstarhat_model(model1,
        #                               train_dataloader1,
        #                               criterion=criterion,
        #                               optimizer=optimizer1,
        #                               epochs=args.epochs)
        # model2 = train_gstarhat_model(model2,
        #                               train_dataloader2,
        #                               criterion=criterion,
        #                               optimizer=optimizer2,
        #                               epochs=args.epochs)

        # DD = np.zeros((2, 2))
        # tem = np.zeros((2, 1))
        # for t in range(0, args.n_timepoints):
        #     for i in range(0, args.n_nodes):
        #         tem[0, 0] = XX_tensor[t, i, 1] - model1(XX_tensor[t, i,
        #                                                           3:8].cuda())
        #         tem[1, 0] = XX_tensor[t, i, 2] - model2(XX_tensor[t, i,
        #                                                           3:8].cuda())
        #         DD = DD + np.dot(tem, tem.T)
        # logger.info(f'MSE: {MSE / args.n_nodes / args.n_timepoints}, loss: {loss.item()}')

        # Ihat = DD / args.n_nodes / args.n_timepoints / (
        #     loss.cpu().detach().numpy())
        # sig2our = np.linalg.inv(Ihat) / args.n_nodes / args.n_timepoints

        # if model.beta1 >= beta[1] - 1.96 * np.sqrt(
        #         sig2our[0, 0]) and model.beta1 <= beta[1] + 1.96 * np.sqrt(
        #             sig2our[0, 0]):
        #     coverage_probability_count[0, 1] += 1
        # if model.beta2 >= beta[2] - 1.96 * np.sqrt(
        #         sig2our[1, 1]) and model.beta2 <= beta[2] + 1.96 * np.sqrt(
        #             sig2our[1, 1]):
        #     coverage_probability_count[0, 2] += 1
        # logger.info(f'Beta1: {model.beta1.item()}; Beta2: {model.beta2.item()}')
        # logger.info(f'Simu: {simu} with Cpcout: {json.dumps((coverage_probability_count/simu).tolist())}')
    # logger.info(
    #     f'Final Cpcout: {json.dumps(coverage_probability_count.tolist())}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = get_training_parser(parser)
    args = parser.parse_args()

    args.n_nodes = 200

    # convert time to string
    time_string = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.work_dir = os.path.join(args.work_dir, time_string)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    logger = Logger.get_logger(name=args.logger_name,
                               file_name=os.path.join(args.work_dir,
                                                      'log.txt'))

    mse_list = []
    for _ in range(200):
        mse = train(args)
        mse_list.append(mse)
    logger.info(f'Average MSE: {np.mean(mse_list)}, std: {np.std(mse_list)}')