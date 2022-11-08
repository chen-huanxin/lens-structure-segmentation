import os
import logging
import cv2
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Union

class Logger(): # remember to close

    def __init__(self, path: str, file_name: str='log.txt') -> None:
        self._path = path
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self._logger = logging.getLogger('train_log_file')    # set logger name
        self._logger.setLevel(logging.INFO)                   # set logger level
        fh_stream = logging.StreamHandler()                         # console handle
        fh_file = logging.FileHandler(os.path.join(self._path, file_name), encoding='utf-8') # log file handle

        fh_stream.setLevel(logging.DEBUG)
        fh_file.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh_stream.setFormatter(formatter)
        fh_file.setFormatter(formatter)

        self._logger.addHandler(fh_stream)
        self._logger.addHandler(fh_file)

        self._visual = SummaryWriter(self._path)
        self._train_log_fh = open(os.path.join(self._path, 'train_log.txt'), 'w')
        self._test_log_fh = open(os.path.join(self._path, 'test_log.txt'), 'w')


    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def addImage(self, tag, img, step, dataformats) -> None:
        self._visual.add_image(tag, img, step, dataformats=dataformats)

    def addImages(self, tag, imgs, step, dataformats) -> None:
        self._visual.add_images(tag, imgs, step, dataformats=dataformats)
    
    def addScalar(self, tag, scalar, step) -> None:
        self._visual.add_scalar(tag, scalar, step)

    # def record(self, phase: str, epoch: int, num: int, output: Tensor, loss: Tensor, step: int, mask: Union[Tensor, None]) -> None:
    #     log_str = f'{phase}_Epoch {epoch} - No.{num}, loss: {loss.item()}'
    #     self._logger.info(log_str)

    #     # 待补充完整

    def recordTrain( self, 
                epoch: int, 
                num: int, 
                loss: float, 
                mean_iu: float, 
                pixel_accuracy: float, 
                mean_accuracy: float, 
                class_accuracy: float, 
                mean_iu0: float, 
                mean_iu1: float,
                step: int):
        phase = 'train'
        # write file
        file_log_str = "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                epoch, num, loss, mean_iu, pixel_accuracy, mean_accuracy, class_accuracy, mean_iu0, mean_iu1)
        fh = self._train_log_fh
        fh.writelines(file_log_str + '\n')

        # write log
        log_str = '{:s} epoch_batch: {:d}_{:d} | loss: {:.4f}  | meanIU: {:.4f} | pixelAccuracy: {:.4f} | meanAccuracy: {:.4f} | classAccuracy: {:.4f} | meanIU0: {:.4f} | meanIU1: {:.4f}'.format(
                        phase, epoch, num, loss, mean_iu, pixel_accuracy, mean_accuracy, class_accuracy, mean_iu0, mean_iu1)
        self._logger.info(log_str)

        # write tensorboard        
        self.addScalar(phase+'_loss', loss, step)
        self.addScalar(phase+'_mean_iu', mean_iu, step)
        self.addScalar(phase+'_pixel_accuracy', pixel_accuracy, step)
        self.addScalar(phase+'_mean_accuracy', mean_accuracy, step)
        self.addScalar(phase+'_class_accuracy', class_accuracy, step)

    def recordTest(self, epoch, num, img_path, mean_iu, pixel_accuracy, mean_accuracy, class_accuracy, final_diff, mean_iu0, mean_iu1):
        log_str = "%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
                    img_path, mean_iu, pixel_accuracy, mean_accuracy, class_accuracy, final_diff, mean_iu0, mean_iu1)
        self._test_log_fh.writelines(log_str + '\n')
        self._logger.info(f'{epoch}_{num}: ', log_str)

    def summary(self, mean_iu: float, pixel_accuracy: float, mean_accuracy: float, class_accuracy: float, final_diff_all: float, mean_iu0: float, mean_iu1: float):
        log_str = "meanIU:%.4f\tpixelAccuracy:%.4f\tmeanAccuracy:%.4f\tclassAccuracy:%.4f\tFinalDifferenceMean:%.4f\tmeanIU0:%.4f\tmeanIU1:%.4f\t" % (mean_iu, pixel_accuracy, mean_accuracy, class_accuracy, final_diff_all, mean_iu0, mean_iu1)
        self._logger.info(log_str)
        self._test_log_fh.writelines(log_str + '\n')

    def saveImg(self, img: np.ndarray, name: str):
        imgs_dir = os.path.join(self._path, 'imgs')
        if not os.path.exists(imgs_dir):
            os.mkdir(imgs_dir)
        cv2.imwrite(os.path.join(imgs_dir, name), img)

    def saveModel(self, model, epoch: int):
        models_dir = os.path.join(self._path, 'models')
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        torch.save(model.state_dict(), 'epoch_'+ str(epoch) + '.pth')

    def close(self):
        self._visual.close()
        self._train_log_fh.close()
        self._test_log_fh.close()
