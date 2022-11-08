import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import _pickle as cPickle
# import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable

class ShapePriorBase(object):
    def __init__(self):
        self.FileName = ''
        self.ShapePriors = []

    def SetFileName(self,FileName_):
        self.FileName = FileName_

    def GetShapePrior(self):
        ShapeP = self.readOneLevelSet(self.FileName)
        return ShapeP

    def readOneLevelSet(self, FileName):
        with open(FileName, 'r+') as f:
            data = cPickle.load(f)
            return data

    #Shape1 and Shape2 are images and all in 0-1
    def ShapeDifference(self,Shape_Pre,Shape_Label):
        Diff = np.sum(Shape_Pre+Shape_Label)
        return Diff

    def FindtheBestFitShape(self, Shape_Pre):
        AllDiffs = []
        for i in range(len(self.ShapePriors)):
            Diffs = self.ShapeDifference(Shape_Pre,self.ShapePriors[i])
            AllDiffs.append(Diffs)

        return 1

    def ShowLevelSet(self, graph):
        fig = plt.figure()
        ax = Axes3D(fig)
        # X, Y value
        X = np.arange(0, graph.shape[0], 1)
        Y = np.arange(0, graph.shape[1], 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, graph, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        plt.show()

class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(20, 30, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        TestImage = torch.randint(low=0, high=1, size=[1,1,512,512], dtype=torch.float32)
        #print(TestOutput)
        SizeOutput = self.localization(TestImage)
        self.LocSize = SizeOutput.size()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.LocSize[1] * self.LocSize[2] * self.LocSize[3], 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.LocSize[1] * self.LocSize[2] * self.LocSize[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # Perform the usual forward pass
        return x

class ShapePriorNet(nn.Module):
    def __init__(self):
        super(ShapePriorNet, self).__init__()
        # The shape prior model
        self.SpatialTransformNet = STNNet()
        self.ShapePriorItem = ShapePriorBase()
        self.gpu_num = 0
        self.ShapeOptions = 2
        self.ShapeTemplateName = ''
        self.show_transform_image = 1
        self.e_ls = 1.0/32

    def BuildShapeModel(self, Options=2):
        LevelSet = self.ShapePriorItem.readOneLevelSet(self.ShapeTemplateName)
        if Options == 1:
            LevelSet = torch.from_numpy(LevelSet)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)

        if Options == 2:
            LevelSet_ = np.zeros_like(LevelSet)
            LevelSet_[LevelSet > 0] = 1
            LevelSet_[LevelSet < 0] = 0
            LevelSet = torch.from_numpy(LevelSet_)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)

        LevelSet = Variable(LevelSet).cuda(self.gpu_num).float()
        return LevelSet

    def PutOnGpu(self, gpu_num):
        self.SpatialTransformNet.cuda(self.gpu_num)

    def SetOptions(self, dic_options):
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.ShapeOptions = dic_options.get('ShapeOptions', self.ShapeOptions)
        self.ShapeTemplateName = dic_options.get('ShapeTemplateName', self.ShapeTemplateName)
        self.e_ls = dic_options.get('e_ls', self.e_ls)
        self.PutOnGpu(self.gpu_num)
        return 1

    def HeavisideFunction(self,FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def forward(self, LevelSetFunction):
        # Add STN, to perform transformation and scale.
        #show the image before transform
        if self.show_transform_image == 1:
            PredictedImage = LevelSetFunction.data.cpu().numpy()
            PredictedImage = np.squeeze(PredictedImage)

        preNum = LevelSetFunction.size()[2] * LevelSetFunction.size()[3]
        # Shape option = 1 : The shape template use the levelset presentation H(phi())
        # Shape option = 2 : The shape template use the 0-1 number
        STNOutPut = self.SpatialTransformNet(LevelSetFunction)
        # Show the transformed Shape
        if self.show_transform_image == 1:
            transformedOut = STNOutPut.data.cpu().numpy()

        # Get the corresponding shape:
        ShapeTemplate = self.BuildShapeModel(Options=self.ShapeOptions)

        # Shape Prior
        ShapeItem1 = self.HeavisideFunction(STNOutPut)
        if self.ShapeOptions == 1:
            ShapeItem2 = self.HeavisideFunction(ShapeTemplate)
        if self.ShapeOptions == 2:
            ShapeItem2 = ShapeTemplate

        if self.show_transform_image == 1:
            ShapeTemplate_image = ShapeTemplate.data.cpu().numpy()

        Loss_Shape_1 = ShapeItem1 - ShapeItem2
        Loss_Shape_2 = torch.abs(Loss_Shape_1)
        Loss_Shape_3 = torch.sum(Loss_Shape_2) / preNum
        Loss_Shape = Loss_Shape_3
        # print('Loss Shape=%f' % Loss_Shape.data.cpu().numpy())
        return Loss_Shape