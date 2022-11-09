from abc import ABCMeta, abstractmethod

import numpy as np
import cv2
from skimage import segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn import Module

from parser_config import Parameters
from logger import Logger
from LevelSets.LevelSetCNN_RNN_STN import LevelSet_CNN_RNN_STN

class Model():
    __metaclass__ = ABCMeta

    def __init__(self, 
                 train_dataset,
                 test_dataset,
                 network: Module,
                 lossfunc,
                 optimizer: Optimizer,
                 logger: Logger,
                 params: Parameters):
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._network = network
        self._lossfunc = lossfunc
        self._optimizer = optimizer
        self._logger = logger
        self._params = params

        # init gpu
        self._device = self.initGPU(self._params.gpu_num)
        self._network.to(self._device)

    
    def initGPU(self, device_num):
        if torch.cuda.is_available():
            torch.cuda.set_device(device_num)
            self._logger.info(f'using cuda: {torch.cuda.get_device_name(device_num)}')
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        return device

    def saveGPUInfo(self):
        self._logger.info('\n' + torch.cuda.memory_summary(self._device))

    def saveModel(self):
        self._logger.saveModel(self._network, self._params.n_epochs)

    def learningRateDecay(self, epoch, lr):
        if (epoch % self._params.lr_decay_epoch == 0 and epoch != 0):
            lr /= 10
            self._optimizer = torch.optim.Adagrad(self._network.parameters(), lr)

        return lr

    def calculateAccuracy(self, confusion):
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

    def evaluate(self, ppi, gt):
        gt_1d = gt.reshape([-1])
        ppi_1d = ppi.reshape([-1])

        confusion = np.zeros([self._params.n_class, self._params.n_class])
        for idx in range(len(gt_1d)):
            confusion[gt_1d[idx], ppi_1d[idx]] += 1
        
        return self.calculateAccuracy(confusion), confusion

    def getDiff(self, label, ppi):
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

        return final_diff

    @abstractmethod
    def trainOnce(self, img, label, tag):
        print('Please define trainOnce()')
        # return ppi, loss
        pass

    @abstractmethod
    def testOnce(self, img, label, tag):
        print('Please define testOnce()')
        # return ppi
        pass

    def train(self, run_test_func=None):
        lr = self._params.lr

        for epoch in range(self._params.n_epochs + 1):
            self._network.train()
            lr = self.learningRateDecay(epoch, lr)

            for idx in range(len(self._train_dataset)):
                img, label, gt, tag = self._train_dataset[idx]
                ppi, loss = self.trainOnce(img, label, tag)

                (meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1), __ = self.evaluate(ppi, gt)
                self._logger.recordTrain(epoch, idx, loss.item(), meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1, epoch*len(self._train_dataset)+idx)

            if run_test_func is not None:
                run_test_func(epoch)

    def test(self, epoch: int=0):
        self._logger.info("====== val ======")
        self._logger.info("====== val ======")
        self._logger.info("====== val ======")

        final_diff_all = 0
        cnt = 0
        confusion_all = np.zeros([self._params.n_class, self._params.n_class])

        with torch.no_grad():
            for idx in range(len(self._test_dataset)):
                img, label, gt, tag = self._test_dataset[idx]
                ppi = self.testOnce(img, label, tag)

                self._logger.saveImg(ppi * 255, str(epoch) + '_' + tag + '.jpg')        # save to host
                self._logger.addImage(tag, ppi * 255, epoch*len(self._test_dataset)+idx, 'HW') # add to tensorboard

                (meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1), confusion = self.evaluate(ppi, gt)
                confusion_all += confusion

                final_diff = self.getDiff(label.numpy(), ppi)
                final_diff_all += final_diff

                self._logger.recordTest(epoch, idx, tag, meanIU, pixelAccuracy, meanAccuracy, classAccuracy, final_diff, meanIU0, meanIU1)
                cnt += 1

        final_diff_mean = final_diff_all / float(cnt)
        meanIU, pixelAccuracy, meanAccuracy, classAccuracy, meanIU0, meanIU1 = self.calculateAccuracy(confusion_all)
        self._logger.summary(meanIU, pixelAccuracy, meanAccuracy, classAccuracy, final_diff_mean, meanIU0, meanIU1)


class ProposedModel(Model):

    def __init__(self, 
                 train_dataset, 
                 test_dataset, 
                 network: Module, 
                 lossfunc, 
                 optimizer: Optimizer, 
                 logger: Logger, 
                 params: Parameters):
        super().__init__(train_dataset, test_dataset, network, lossfunc, optimizer, logger, params)

        # init activate function
        if self._params.UseSigmoid:
            self._sigmoid = nn.Sigmoid()
            self._activate_func = lambda out : (self._sigmoid(out) - 0.5) * 2
        else:
            self._activate_func = nn.Tanh()

        self._level_set_model = LevelSet_CNN_RNN_STN()
        self._level_set_model.SetOptions(self._params.getDict())


    def trainOnce(self, img, label, tag):
        self._network.zero_grad()
        img_gpu = img.to(self._device)
        label_gpu = label.to(self._device)
        out_gpu = self._network(img_gpu)

        out_gpu = F.upsample_nearest(out_gpu, [512, 512])
        out_gpu = self._activate_func(out_gpu)

        target_out = torch.split(out_gpu, split_size_or_sections=1, dim=1)[1]
        target_img = torch.split(img_gpu, split_size_or_sections=1, dim=1)[1]

        level_set_label = self._level_set_model.GetLevelSetFile(tag)
        loss = self._level_set_model.Train(target_img, target_out, label_gpu, level_set_label)

        loss.backward()
        self._optimizer.step()

        out_np = out_gpu.cpu().detach().numpy()
        out_np = np.squeeze(out_np)
        out_np = np.where(out_np > 0, 1, 0)
        ppi = out_np.astype(np.uint8)

        return ppi, loss

    def testOnce(self, img, label, tag):
        img_gpu = img.to(self._device)
        out_gpu = self._network(img_gpu)
        out_gpu = F.upsample_nearest(out_gpu, [512, 512])

        target_out = torch.split(out_gpu, split_size_or_sections=1, dim=1)[1]

        out_np = target_out.cpu().detach().numpy() 
        out_np = np.squeeze(out_np)
        out_np = np.where(out_np > 0, 1, 0)
        ppi = out_np.astype(np.uint8)
        
        return ppi


class BaselineModel(Model):

    def __init__(self, 
                 train_dataset, 
                 test_dataset, 
                 network: Module, 
                 lossfunc, 
                 optimizer: Optimizer, 
                 logger: Logger, 
                 params: Parameters):
        super().__init__(train_dataset, test_dataset, network, lossfunc, optimizer, logger, params)
        self._activate_func = nn.Softmax2d()

    def trainOnce(self, img, label, tag):
        self._network.zero_grad()
        img_gpu = img.to(self._device)
        label_gpu = label.to(self._device)

        out_gpu = self._network(img_gpu)
        loss = self._lossfunc(out_gpu, label_gpu)
        ppi = np.argmax(out_gpu.cpu().detach().numpy(), 1).reshape((self._params.img_size, self._params.img_size))
        
        loss.backward()
        self._optimizer.step()

        return ppi, loss

    def testOnce(self, img, label, tag):
        img_gpu = img.to(self._device)
        out_gpu = self._network(img_gpu)
        out_gpu = torch.log(self._activate_func(out_gpu))
        ppi = np.argmax(out_gpu.cpu().detach().numpy(), 1).reshape((self._params.img_size, self._params.img_size))
        
        return ppi