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
import cv2
import numpy as np

import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage import segmentation

from parser_config import Parameters
from dataset import LensDataset
from unet import UNet1024
from logger import Logger 
from LevelSets.LevelSetCNN_RNN_STN import LevelSet_CNN_RNN_STN

def calculate_Accuracy(confusion):
    confusion = np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU) # 此处从IU改成了IU[1]!!!又改成了IU[0]试试，结果IOU可能会变低
    meanIU0 = np.mean(IU[0])
    meanIU1 = np.mean(IU[1])
    pos[pos == 0] = 1
    res[res == 0] = 1
    pixelAccuracy = np.sum(tp) / np.sum(confusion)
    meanAccuracy = np.mean(tp / pos)
    classAccuracy = np.mean(tp / res)
    return  meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1

def get_parse():
    parser = argparse.ArgumentParser(description='Level set lens segmentation')
    parser.add_argument('--ini_path', '-i', type=str, default='config.ini',
                        help='the path of configuration file')
    parser.add_argument('--model', '-m', type=str, default='Proposed', choices=['UNet', 'Proposed'],
                        help='the model name')
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
    net = UNet1024([3, 1024, 1024])
    try:
        net.cuda(params.gpu_num)
    except:
        print('failed to get gpu')

    if params.IfFineTune == 1:
        fine_tune_dir = params.FineTuneModelDIr
        net.load_state_dict(torch.load(fine_tune_dir))

    level_set_model = LevelSet_CNN_RNN_STN()
    level_set_model.SetOptions(params.getDict())

    # init dataset
    dataset_dir = params.DataSetDir
    train_dataset = LensDataset(dataset_dir, img_size=params.img_size)
    # train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    test_dataset = LensDataset(dataset_dir, img_size=params.img_size, train_flag=False)
    # test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # init activate function
    if params.UseSigmoid:
        sigmoid = nn.Sigmoid()
        activate_func = lambda out : (sigmoid(out) - 0.5) * 2 # 此处需要测试能不能使用
    else:
        activate_func = nn.Tanh()

    # init optim
    lr = params.lr
    optimizer = torch.optim.Adagrad(net.parameters(), lr)

    for epoch in range(params.n_epochs + 1):
        # train
        net.train()
        if epoch % params.lr_decay_epoch == 0 and epoch != 0:
            lr /= 10
            optimizer = torch.optim.Adagrad(net.parameters(), lr)

        num = 0
        # for img, label, gt, tag in train_dataset: # will overflow
        for index in range(len(train_dataset)):
            img, label, gt, tag = train_dataset[index]
            net.zero_grad()
            img_gpu = img.to(params.gpu_num)
            label_gpu = label.to(params.gpu_num)
            out = net(img_gpu)
            out = F.upsample_nearest(out, [512, 512])

            out = activate_func(out)
            target_out = torch.split(out, split_size_or_sections=1, dim=1)[1]
            target_img = torch.split(img_gpu, split_size_or_sections=1, dim=1)[1]

            level_set_label = level_set_model.GetLevelSetFile(tag)
            loss = level_set_model.Train(target_img, target_out, label_gpu, level_set_label)

            img_np = target_img.cpu().numpy()
            label_np = label.numpy()
            label_np = np.squeeze(label_np).astype(np.uint8)

            out_np = out.cpu().detach().numpy() # out原本应该是在gpu上的
            out_np = np.squeeze(out_np)
            out_np = np.where(out_np > 0, 1, 0)
            ppi = out_np.astype(np.uint8)

            loss.backward()

            gt_1d = gt.reshape([-1])
            out_1d = ppi.reshape([-1])

            confusion = np.zeros([params.n_class, params.n_class])
            for idx in range(len(gt_1d)):
                confusion[gt_1d[idx], out_1d[idx]] += 1
            
            meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1 = calculate_Accuracy(confusion)
            logger.recordTrain(epoch, num, loss.item(), meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1, epoch*len(train_dataset)+num)

            num += 1
            optimizer.step()

        # test every 2 epochs
        if epoch % 2 == 1:
            logger.info("====== val ======")
            logger.info("====== val ======")
            logger.info("====== val ======")

            cnt = 0
            final_diff_all = 0
            confusion_all = np.zeros([params.n_class, params.n_class])

            net.eval()
            # for img, label, gt, tag in test_dataset: # will overflow
            for index in range(len(test_dataset)):
                img , label, gt, tag = test_dataset[index]
                net.zero_grad()
                img_gpu = img.to(params.gpu_num)

                out = net(img_gpu)
                out = F.upsample_nearest(out, [512, 512])
                target_out = torch.split(out, split_size_or_sections=1, dim=1)[1]
                target_img = torch.split(img_gpu, split_size_or_sections=1, dim=1)[1]

                img_np = target_img.cpu().numpy() # 这里不知道会不会有问题，原来的写法是target_img.data.numpy()
                out_np = out.cpu().detach().numpy() 
                out_np = np.squeeze(out_np)
                out_np = np.where(out_np > 0, 1, 0)
                ppi = out_np.astype(np.uint8)

                logger.saveImg(ppi * 255, str(epoch) + '_' + tag + '.jpg')
                
                # calculate nucleus differenct:
                label_level_set = label
                label_level_set = np.squeeze(label_level_set).astype(np.uint8)
                invers_label = 1 - label_level_set
                transformed_img_top_zero = cv2.distanceTransform(label_level_set, cv2.DIST_L2, 5)
                transformed_img_down_zero = cv2.distanceTransform(invers_label, cv2.DIST_L2, 5)

                gt_distance = transformed_img_top_zero + transformed_img_down_zero

                test_pre = np.squeeze(ppi).astype(np.uint8)
                # find the top_down lines
                pre_boundarys = segmentation.find_boundaries(test_pre, mode='inner')

                # calculate full difference
                point_num = np.sum(pre_boundarys) + 1
                final_diff = np.sum(gt_distance * pre_boundarys) / point_num
                final_diff_all += final_diff
                cnt += 1
                
                # write log
                gt_1d = label.reshape([-1])
                out_1d = ppi.reshape([-1])
                confusion = np.zeros([params.n_class, params.n_class])

                for idx in range(len(gt_1d)):
                    confusion[gt_1d[idx], out_1d[idx]] += 1
                    confusion_all[gt_1d[idx], out_1d[idx]] += 1

                meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1 = calculate_Accuracy(confusion)
                logger.recordTest(epoch, idx, tag, meanIU, pixelAccuracy, meanAccuracy, classAccuracy, final_diff, meanIU0, meanIU1)

            final_diff_mean = final_diff_all / float(cnt)
            meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1 = calculate_Accuracy(confusion_all)
            logger.summary(meanIU, pixelAccuracy, meanAccuracy, classAccuracy, final_diff_mean, meanIU0, meanIU1)

    logger.saveModel(net, params.n_epochs)
    logger.info('success')
    logger.close()

