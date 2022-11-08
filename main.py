"""
#
# @brief: train model
# @time: created on 2022-11-06 15:11:52
# @author: 
# @coding: utf-8
# @version: V1.1.1

"""

import argparse
import os

import torch
import torch.nn as nn

from parser_config import Parameters
from dataset import LensDataset
from logger import Logger
from model import ProposedModel, BaselineModel

from networks.unet import UNet1024
from networks.mnet import M_Net
from networks.fpn import FPN50
from networks.pspnet import PSPNet

def get_parse():
    parser = argparse.ArgumentParser(description='Level set lens segmentation')
    parser.add_argument('--ini_path', '-i', type=str, default='config.ini',
                        help='the path of configuration file')
    parser.add_argument('--model', '-m', type=str, default='Proposed', choices=['Baseline', 'Proposed'],
                        help='the model name')
    parser.add_argument('--network', '-n', type=str, default='UNet1024', choices=['UNet1024', 'PSPNet', 'MNet', 'FPN50'],
                        help='neural network: UNet1024, PSPNet, MNet, FPN50')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # init parameters
    args = get_parse()
    params = Parameters(args.ini_path) # read configuration file

    # init logger
    log_dir = os.path.join(params.LogDir, params.ModelName)
    logger = Logger(log_dir)

    # init neural network
    if args.network == 'UNet1024':
        net = UNet1024([3, 1024, 1024])
    elif args.network == 'PSPNet':
        net = PSPNet(pretrained=False)
    elif args.network == 'MNet':
        net = M_Net(n_classes=2)
    else:
        net = FPN50()
        
    try:
        net.cuda(params.gpu_num)
    except:
        print('failed to get gpu')

    if params.IfFineTune == 1:
        fine_tune_dir = params.FineTuneModelDIr
        net.load_state_dict(torch.load(fine_tune_dir))

    # init dataset
    dataset_dir = params.DataSetDir
    train_dataset = LensDataset(dataset_dir, img_size=params.img_size)
    # train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    test_dataset = LensDataset(dataset_dir, img_size=params.img_size, train_flag=False)
    # test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # init loss function
    lossfunc = nn.NLLLoss2d()

    # init optim
    lr = params.lr
    optimizer = torch.optim.Adagrad(net.parameters(), lr)

    # init model
    if args.model == 'Proposed':
        model = ProposedModel(train_dataset, test_dataset, net, lossfunc, optimizer, logger, params)
    else:
        model = BaselineModel(train_dataset, test_dataset, net, lossfunc, optimizer, logger, params)

    # define test function
    def func(epoch):
        if epoch % 2:
            model.test(epoch // 2)

    model.train(func) # test during training

    model.saveModel()
    model.saveGPUInfo()
    
    logger.info('success')
    logger.close()