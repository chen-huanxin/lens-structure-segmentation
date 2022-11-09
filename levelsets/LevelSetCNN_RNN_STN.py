import torch
from torch.autograd import Variable
from skimage import transform
import numpy as np
import os

from levelsets.LevelSetLoss import LevelSet_Loss
from levelsets.STNLoss import ShapePriorNet
from levelsets.GRU2DLoss import GRU2D

class LevelSet_CNN_RNN_STN(object):
    def __init__(self):
        self.LevelSetModel = LevelSet_Loss()
        self.ShapePriorModel = ShapePriorNet()
        self.GRUEvolutionModel = GRU2D()
        self.LevelSetLabelDir = ''
        self.gpu_num = 0
        self.inputSize = (512,512)
        self.NormalizationOption = 0
        self.NormalizationOption_3_Alpha = 30
        self.NormalizationOption_3_Beta = 30

    def SetOptions(self,Option):
        self.Options = Option
        self.LevelSetModel.SetOptions(self.Options)
        self.ShapePriorModel.SetOptions(self.Options)
        self.GRUEvolutionModel.SetOptions(self.Options)
        self.LevelSetLabelDir = Option.get('LevelSetLabelDir',self.LevelSetLabelDir)
        self.gpu_num = Option.get('gpu_num', self.gpu_num)
        self.inputSize = Option.get('inputSize', self.inputSize)
        self.NormalizationOption = Option.get('NormalizationOption', self.NormalizationOption)
        self.NormalizationOption_3_Alpha = Option.get('NormalizationOption_3_Alpha', self.NormalizationOption_3_Alpha)
        self.NormalizationOption_3_Beta= Option.get('NormalizationOption_3_Beta', self.NormalizationOption_3_Beta)

    def Train(self, Image_, OutPut, Label_, LevelSetLabel = 0):
        LevelSets_ = self.LevelSetModel.OutputLevelSet(FeatureMap=OutPut)
        loss_CNN = 0
        lossShape = 0
        lossRNN = 0
        if self.Options['CNNEvolution'] == 1:
            loss_CNN = self.LevelSetModel.LevelSetLoss(Image_=Image_, OutPut_FeatureMap = OutPut, LabelMap = Label_, LevelSetLabel = LevelSetLabel)
        if self.Options['ShapePrior'] == 1:
            lossShape = self.ShapePriorModel(LevelSets_)
        if self.Options['RNNEvolution'] == 1:
            self.OutLevelSetFunction = self.GRUEvolutionModel.ForwardRNN2D(LevelSets_, Image_)
            lossRNN = self.GRUEvolutionModel.RNNLoss(Image_ = Image_, LevelSetFunction = self.OutLevelSetFunction, LabelMap = Label_)
        loss = self.Options['lambda_CNN' ] * loss_CNN + self.Options['lambda_shape' ] * lossShape + self.Options['Lamda_RNN' ] * lossRNN
        return loss

    def GetLevelSetFile(self, Image_name):
        levelSet_Label_Path = os.path.join(self.LevelSetLabelDir, Image_name)
        LevelSetLabel_Name = levelSet_Label_Path + '.npy'

        LevelSetLabel = np.load(LevelSetLabel_Name)

        Final = LevelSetLabel / 1000
        Final_1 = transform.resize(Final, self.inputSize)
        Final_LevelSet_Label = Final_1 * 1000

        # Normalization
        Final_LevelSet_Label = self.Normalization(Final_LevelSet_Label)
        Final_LevelSet_Label = Final_LevelSet_Label[np.newaxis, :, :]
        LevelSetLabel = Variable(torch.from_numpy(Final_LevelSet_Label)).float().cuda(self.gpu_num)
        return LevelSetLabel

    def Normalization(self,featureMap):
        MaxDistance = np.max(featureMap)
        MinDistance = np.min(featureMap)
        # Normalize the levelset area to [~,1]
        if self.NormalizationOption == 1:
            Final_LevelSet_Label = featureMap / MaxDistance

        # Normalize the levelset area to [-1,1]
        if self.NormalizationOption == 2:
            featureMap = featureMap - MinDistance
            MaxDistance = np.max(featureMap)
            Final_LevelSet_Label = (featureMap / MaxDistance) * 2
            Final_LevelSet_Label = Final_LevelSet_Label - 1
        # Normalize the levelset through function
        #  f(x)=  { -30 x<a
        #         { 30  x>beta
        if self.NormalizationOption == 3:
            Final_LevelSet_Label = featureMap.copy()
            Final_LevelSet_Label[featureMap > self.NormalizationOption_3_Beta] = self.NormalizationOption_3_Beta
            Final_LevelSet_Label[featureMap < -self.NormalizationOption_3_Alpha] = -self.NormalizationOption_3_Alpha

            MinDistance = np.min(Final_LevelSet_Label)
            #Normalize to [-1,1]
            Final_LevelSet_Label = Final_LevelSet_Label - MinDistance
            MaxDistance = np.max(Final_LevelSet_Label)
            Final_LevelSet_Label = (Final_LevelSet_Label / MaxDistance) * 2
            Final_LevelSet_Label = Final_LevelSet_Label - 1
        return Final_LevelSet_Label

    def LevelSetMask(self, OutPut):
        #First_Level_set
        out = np.squeeze(OutPut)
        out[out > 0] = 1
        out[out < 0] = 0
        #Second_Term: ZeroLevelSet
        #CNN_Pre = self.LevelSetModel.LevelSetMask(OutPut)
        #return CNN_Pre
        return out

    def PreRNN(self):
        return self.OutLevelSetFunction

    def PreCNN_RNN(self):
        return 1

    def DefaultOptions(self):
        Options = {
            'InnerAreaOption': 2,
            'UseLengthItemType': 1,
            'UseHigh_Hfuntion': 0,
            'isShownVisdom': 0,
            'lambda_1': 0.75,
            'lambda_2': 0.01,
            'lambda_3': 0.01,
            'lambda_shape': 0.0,
            'lambda_CNN': 0.75,
            'Lamda_RNN': 1.0,
            'GRU_Number': 1,
            'RNNEvolution': 1,
            'ShapePrior': 0,
            'inputSize': (512, 512),
            'ShapeTemplateName': '/home/intern1/guanghuixu/resnet/shapePrior/1390_L_004_110345.pkl',
            'gpu_num': 0,
            'CNNEvolution': 0,
            'GRU_Dimention': 2,
            'NormalizationOption':1,
            'NormalizationOption_3_Alpha':30,
            'NormalizationOption_3_Beta':30,
        }
        self.SetOptions(Options)