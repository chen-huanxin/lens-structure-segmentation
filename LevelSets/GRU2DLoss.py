import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class GRU2DCell(nn.Module):
    def __init__(self):
        super(GRU2DCell, self).__init__()
        #For z
        self.z_Item_Uz = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.z_Item_Wz = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.r_Item_Ur = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.r_Item_Wr = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.h_Item_Uh = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.h_Item_Wh = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.Matrix_U_g = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.Matrix_W_g = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        self.sigmoid_  = nn.Sigmoid()
        self.tanh_     = nn.Tanh()
        self.forwardNum = 5
        self.gpu_num = 1
        self.e_ls = 1.0/32

        self.Sobelx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        self.Sobely = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        WeightX = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        WeightY = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        WeightX = WeightX[np.newaxis, np.newaxis, :, :]
        WeightY = WeightY[np.newaxis, np.newaxis, :, :]
        WeightX = torch.FloatTensor(WeightX)
        WeightY = torch.FloatTensor(WeightY)
        self.Sobelx.weight.data = WeightX
        self.Sobely.weight.data = WeightY
        self.Sobelx.weight.requires_grad = False
        self.Sobely.weight.requires_grad = False

        self.Dif_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xx_weight = np.asarray([[0, 0, 0], [1, -2, 1], [0, 0, 0]])

        self.Dif_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_yy_weight = np.asarray([[0, 1, 0], [0, -2, 0], [0, -1, 0]])

        self.Dif_xy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xy_weight = np.asarray([[0, -1, 1], [0, 1, -1], [0, 0, 0]])

        Dif_xx_weight = Dif_xx_weight[np.newaxis, np.newaxis, :, :]
        Dif_yy_weight = Dif_yy_weight[np.newaxis, np.newaxis, :, :]
        Dif_xy_weight = Dif_xy_weight[np.newaxis, np.newaxis, :, :]

        Dif_xx_weight = torch.FloatTensor(Dif_xx_weight)
        Dif_yy_weight = torch.FloatTensor(Dif_yy_weight)
        Dif_xy_weight = torch.FloatTensor(Dif_xy_weight)

        self.Dif_xx.weight.data = Dif_xx_weight
        self.Dif_yy.weight.data = Dif_yy_weight
        self.Dif_xy.weight.data = Dif_xy_weight

        self.Dif_xx.weight.requires_grad = False
        self.Dif_yy.weight.requires_grad = False
        self.Dif_xy.weight.requires_grad = False

    def SetOptions(self, dic_options):
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.putonGPU()
        return 1

    def putonGPU(self):
        self.z_Item_Uz.cuda(self.gpu_num)
        self.z_Item_Wz.cuda(self.gpu_num)
        self.r_Item_Ur.cuda(self.gpu_num)
        self.r_Item_Wr.cuda(self.gpu_num)
        self.h_Item_Uh.cuda(self.gpu_num)
        self.h_Item_Wh.cuda(self.gpu_num)
        self.Matrix_U_g.cuda(self.gpu_num)
        self.Matrix_W_g.cuda(self.gpu_num)

        self.Sobelx.cuda(self.gpu_num)
        self.Sobely.cuda(self.gpu_num)
        self.Dif_xx.cuda(self.gpu_num)
        self.Dif_yy.cuda(self.gpu_num)
        self.Dif_xy.cuda(self.gpu_num)

    #Calculate the curvature of the level set function
    def GetCurvature(self, Phi_t0):
        a = Phi_t0.data.cpu().numpy()
        # Phi_t0 = HeavisideFunction(Phi_t0) #Get Level Set Map
        Item1 = self.Dif_xx(Phi_t0) * torch.pow(self.Sobely(Phi_t0), 2)
        Item2 = 2 * self.Sobelx(Phi_t0) * self.Sobely(Phi_t0) * self.Dif_xy(Phi_t0)
        Item3 = self.Dif_yy(Phi_t0) * torch.pow(self.Sobelx(Phi_t0), 2)
        Item4 = torch.pow(self.Sobelx(Phi_t0), 2) + torch.pow(self.Sobely(Phi_t0), 2)

        Item4Values = Item4.data.cpu().numpy()
        Item4Index = np.where(Item4Values == 0)
        ItemMask = np.zeros_like(Item4Values)
        ItemMask[Item4Index] = 1
        ItemMaskTensor = torch.from_numpy(ItemMask)
        ItemMaskTensor = Variable(ItemMaskTensor).cuda(self.gpu_num).float()

        ItemDivide = torch.pow(Item4, 3.0 / 2.0)
        # Prevent Divide by zero
        ItemDivide = ItemDivide + ItemMaskTensor
        ItemAll = (Item1 + Item2 + Item3) / ItemDivide

        return ItemAll

    def GenerateRLSInput(self, Image_, Phi_t0):
        Curvature_ = self.GetCurvature(Phi_t0)
        # U_g(I-c1)^2 + W_g(I-c2)^2
        # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
        # We use the second term
        #C_1, C_2 = self.GetC1_C2(Phi_t0, Image_, Option=2)
        HeavisideLevelSet = self.HeavisideFunction(Phi_t0)
        U0xy = Image_
        C_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
        C_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)

        Item1 = torch.pow(Image_ - C_1, 2)
        Item2 = torch.pow(Image_ - C_2, 2)
        FinalItem = Curvature_ - self.Matrix_U_g(Item1) + self.Matrix_W_g(Item2)

        return FinalItem

    def HeavisideFunction(self,FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def DiracDeltaFunction(self,FeatureMap):
        Output = (1.0 / np.pi) * self.e_ls / (self.e_ls * self.e_ls + torch.pow(FeatureMap, 2))
        return Output

    def forward1(self, LevelSets, Images):
        Input = self.GenerateRLSInput(Image_=Images, Phi_t0=LevelSets)
        Z_ = self.sigmoid_(self.z_Item_Uz(Input) + self.z_Item_Wz(Input))
        R_ = self.sigmoid_(self.r_Item_Ur(Input) + self.r_Item_Wr(Input))
        ht_ = self.tanh_(self.h_Item_Uh(Input) + self.h_Item_Wh(R_ * LevelSets))
        LevelSets = (1-Z_)*ht_ + Z_*LevelSets
        return LevelSets

    def forwardn(self, LevelSets, Images):
        for i in range(self.forwardNum):
            LevelSets = self.forward1(LevelSets, Images)
        return LevelSets

class GRU2D(nn.Module):
    def __init__(self):
        super(GRU2D, self).__init__()
        self.RNNLoss = torch.nn.BCEWithLogitsLoss()
        self.GRU_hiddenSize = 0
        self.GRU_inputSize = 0
        self.GRU_TimeLength = 1
        self.GRU_Dimention = 2
        self.GRU_Number = 0
        self.HasGRU = 0
        self.RNNDimension = 2
        self.gpu_num = 0
        self.inputSize = (512,512)

    def SetOptions(self, dic_options):
        self.GRU_TimeLength = dic_options.get('GRU_TimeLength', self.GRU_TimeLength)
        self.GRU_Dimention = dic_options.get('GRU_Dimention', self.GRU_Dimention)
        self.GRU_Number = dic_options.get('GRU_Number', self.GRU_Number)
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.inputSize = dic_options.get('inputSize', self.inputSize)

        if self.GRU_Dimention ==1:
            self.BuildRNNModle1D()
        if self.GRU_Dimention ==2:
            self.BuildRNNModle2D()

    #Build a RNN module
    def BuildRNNModle1D(self):
        # The GRU level set evolution model
        self.GRU_hiddenSize = self.inputSize[0] * self.inputSize[1]
        self.GRU_inputSize  = self.inputSize[0] * self.inputSize[1]
        self.levelSetEvolutionModel = torch.nn.GRUCell(self.GRU_hiddenSize,self.GRU_inputSize)
        self.levelSetEvolutionModel.cuda(self.gpu_num)
        return self.levelSetEvolutionModel

    def BuildRNNModle2D(self):
        # The GRU level set evolution model
        self.GRU_hiddenSize = self.inputSize
        self.GRU_inputSize = self.inputSize
        self.levelSetEvolutionModel = []
        GRUOptions={
            'gpu_num':self.gpu_num
        }
        for i in range(self.GRU_Number):
            GRUInstance = GRU2DCell()
            GRUInstance.SetOptions(GRUOptions)
            self.levelSetEvolutionModel.append(GRUInstance)
        return 1

    def ForwardRNN2D(self, Init_LevelSetFunction, Image_):
        Levelset = Init_LevelSetFunction
        for i in range(self.GRU_Number):
            selectedModel = self.levelSetEvolutionModel[i]
            Levelset = selectedModel.forwardn(Levelset,Image_)
        return Levelset

    def ForwardRNN1D(self, Init_LevelSetFunction, Image_):
        HiddenLevelSet_1 = self.GenerateRLSInput(Image_=Image_, Phi_t0=Init_LevelSetFunction)
        ImageSize = Image_.size()
        LevelSetFunctionSize = Init_LevelSetFunction.size()
        Image_Input1 = torch.reshape(Image_,[ImageSize[0],-1])

        for i in range(self.GRU_TimeLength):
            HiddenLevelSet_Input1 = torch.reshape(HiddenLevelSet_1, [LevelSetFunctionSize[0], -1])
            HiddenLevelSet_2 = self.levelSetEvolutionModel(Image_Input1, HiddenLevelSet_Input1)
            ReshapeBackLevelSet = torch.reshape(HiddenLevelSet_2,ImageSize)
            HiddenLevelSet_1 = self.GenerateRLSInput(Image_=Image_Input1, Phi_t0=ReshapeBackLevelSet)
        return ReshapeBackLevelSet

    # First step, generate RLS input:
    # x_t = g(I,\phi_t-1)
    def GenerateRLSInput(self, Image_, Phi_t0):
        Curvature_ = self.GetCurvature(Phi_t0)
        # U_g(I-c1)^2 + W_g(I-c2)^2
        # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
        # We use the second term
        # C_1, C_2 = self.GetC1_C2(Phi_t0, Image_, Option=2)
        HeavisideLevelSet = self.HeavisideFunction(Phi_t0)
        U0xy = Image_
        C_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
        C_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)

        Item1 = torch.pow(Image_ - C_1, 2)
        Item2 = torch.pow(Image_ - C_2, 2)
        FinalItem = Curvature_ - self.Matrix_U_g(Item1) + self.Matrix_W_g(Item2)

        return FinalItem

    def forward(self, Image_, LevelSetFunction):
        if self.RNNDimension == 1:
            RNN_Output = self.ForwardRNN1D(LevelSetFunction, Image_)
        if self.RNNDimension == 2:
            RNN_Output = self.ForwardRNN2D(LevelSetFunction, Image_)
        return RNN_Output

    def RNNLoss(self, Image_, LevelSetFunction, LabelMap):
        RNN_Output_ = self.forward(Image_, LevelSetFunction)
        OutputSize = RNN_Output_.size()
        RNN_Output_ = RNN_Output_.view([OutputSize[0], -1])
        LabelMap_ = LabelMap.view([OutputSize[0], -1])
        LossRNN = self.RNNLoss(RNN_Output_, LabelMap_)
        return LossRNN