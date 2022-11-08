import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """

    C = tensor.size(1)  # 获取图像的维度
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        
        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice

class LevelSet_Loss(object):

    def __init__(self):
        self.lambda_1 = 0.75
        self.lambda_2 = 0.005
        self.lambda_3 = 0.00
        self.lambda_shape = 0.001
        self.lambda_rnn = 0
        self.small_e = 0.00001
        self.e_ls = 1.0 / 128.0
        self.Highe_ls = 1.0 / 1024.0
        self.InnerAreaOption = 1
        self.UseLengthItemType = 1
        self.isShownVisdom = 1
        self.ShapePrior = 0
        self.RNNEvolution = 0
        self.CNNEvolution = 1
        self.inputSize = (512,512)
        self.ShapeTemplateName = ''
        self.gpu_num = 0
        self.GRU_hiddenSize = 0
        self.GRU_inputSize = 0
        self.GRU_TimeLength = 1
        self.GRU_Dimention = 2
        self.GRU_Number = 0
        self.HasGRU = 0
        self.UseHigh_Hfuntion = 0  # for length item
        self.PrintLoss = 1
        self.Lamda_LevelSetDifference = 1

        # This is calculate the gradient of the Image
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

        # The initial U_g and W_g
        self.Matrix_U_g = torch.nn.Linear(in_features=1, out_features=1, bias=False)
        self.Matrix_W_g = torch.nn.Linear(in_features=1, out_features=1, bias=False)

        self.softmax_2d = torch.nn.Softmax2d()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu_ = torch.nn.ReLU()

        self.diceloss = DiceLoss()

    def SetPatameter(self,Lamda1=0.75, Lamda2=0.005, Lamda3=0.2, e_ls=1.0 / 128.0):
        self.lambda_1 = Lamda1
        self.lambda_2 = Lamda2
        self.lambda_3 = Lamda3
        self.e_ls = e_ls

    # Transform to [-1,1]
    # sigmoid is [0,1]
    # the output of the feature map is [0,1]
    # transform the output to [-0.5,0.5]
    # Generate the LevelSet Function
    def OutputLevelSet(self, FeatureMap):
        out = self.sigmoid(FeatureMap)
        out = out - 0.5
        return out

    # Generate the LevelSet Mask
    def LevelSetMask(self, FeatureMap):
        out = self.sigmoid(FeatureMap)
        out = out.data.cpu().numpy()
        out = out - 0.5
        out[out > 0] = 1
        out[out < 0] = 0
        return out

    # calculate the c1 and c2 of the level set
    # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
    # Option = 1  U0xy = H(phi(x,y))
    # Option = 2  U0xy = Image
    def GetC1_C2(self, Phi_t0, Image, Option=1):
        SelectedTensor = self.HeavisideFunction(Phi_t0)
        if Option == 1:
            U0xy = self.HeavisideFunction(Phi_t0)
        if Option == 2:
            U0xy = Image
        c_1 = torch.sum(U0xy * SelectedTensor) / torch.sum(SelectedTensor)
        c_2 = torch.sum(U0xy * (1 - SelectedTensor)) / torch.sum(1 - SelectedTensor)
        return c_1, c_2

    # Calculate the curvature of the level set function
    def GetCurvature(self,Phi_t0):
        # Phi_t0 = HeavisideFunction(Phi_t0) #Get Level Set Map
        Item1 = self.Dif_xx(Phi_t0) * torch.pow(self.Sobely(Phi_t0), 2)
        Item2 = 2 * self.Sobelx(Phi_t0) * self.Sobely(Phi_t0) * self.Dif_xy(Phi_t0)
        Item3 = self.Dif_yy(Phi_t0) * torch.pow(self.Sobelx(Phi_t0), 2)
        Item4 = torch.pow(self.Sobelx(Phi_t0), 2) + torch.pow(self.Sobely(Phi_t0), 2)
        ItemAll = (Item1 + Item2 + Item3) / torch.pow(Item4, 3.0 / 2.0)
        return ItemAll

    # Put all the operator on the GPU
    def PutOnGpu(self, gpu_num):
        self.Sobelx.cuda(self.gpu_num)
        self.Sobely.cuda(self.gpu_num)
        self.Dif_xx.cuda(self.gpu_num)
        self.Dif_yy.cuda(self.gpu_num)
        self.Dif_xy.cuda(self.gpu_num)
        self.Matrix_U_g.cuda(self.gpu_num)
        self.Matrix_W_g.cuda(self.gpu_num)
        self.softmax_2d.cuda(self.gpu_num)
        self.sigmoid.cuda(self.gpu_num)
        self.relu_.cuda(self.gpu_num)
        #self.SpatialTransformNet.cuda(self.gpu_num)

    def HeavisideFunction(self,FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def HighElsHeavisideFunction(self, FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.Highe_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def DiracDeltaFunction(self,FeatureMap):
        Output = (1.0 / np.pi) * self.e_ls / (self.e_ls * self.e_ls + torch.pow(FeatureMap, 2))
        return Output

    # InnerAreaOption = 1  U0xy = H(phi(x,y))
    # InnerAreaOption = 2  U0xy = Image
    # UseLengthItemType == 1 : |delta H(x)|
    # UseLengthItemType == 2 : dirace(phi(x,y))|delta phi(x,y)|
    # dic_options is a dictionary
    # Usage:
    # SetOptions({'InnerAreaOption':2,'UseLengthItemType':1,'isShownVisdom':1})
    def SetOptions(self,dic_options):
        self.InnerAreaOption = dic_options.get('InnerAreaOption',self.InnerAreaOption)
        self.UseLengthItemType = dic_options.get('UseLengthItemType',self.UseLengthItemType)
        self.isShownVisdom = dic_options.get('isShownVisdom',self.isShownVisdom)
        self.lambda_1 = dic_options.get('lambda_1',self.lambda_1)
        self.lambda_2 = dic_options.get('lambda_2',self.lambda_2)
        self.lambda_3 = dic_options.get('lambda_3',self.lambda_3)
        self.e_ls = dic_options.get('e_ls',self.e_ls)
        self.RNNEvolution = dic_options.get('RNNEvolution', self.RNNEvolution)
        self.ShapePrior = dic_options.get('ShapePrior', self.ShapePrior)
        self.inputSize = dic_options.get('inputSize', self.inputSize)
        self.ShapeTemplateName = dic_options.get('ShapeTemplateName', self.ShapeTemplateName)
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.CNNEvolution = dic_options.get('CNNEvolution', self.CNNEvolution)
        self.GRU_TimeLength =  dic_options.get('GRU_TimeLength', self.GRU_TimeLength)
        self.lambda_shape = dic_options .get('lambda_shape', self.lambda_shape)
        self.UseHigh_Hfuntion = dic_options.get('UseHigh_Hfuntion', self.UseHigh_Hfuntion)
        self.GRU_Dimention = dic_options.get('GRU_Dimention', self.GRU_Dimention)
        self.lambda_rnn = dic_options.get('Lamda_RNN', self.lambda_rnn)
        self.GRU_Number = dic_options.get('GRU_Number', self.GRU_Number)
        self.Highe_ls = dic_options.get('Highe_ls', self.Highe_ls)
        self.Lamda_LevelSetDifference = dic_options.get('Lamda_LevelSetDifference', self.Lamda_LevelSetDifference)
        self.PutOnGpu(self.gpu_num)

    # FeatureMap is the size [255,255,N]
    # LabelMap is the size [255,255,N]
    # For binary classification, 0-background, 1-foreground
    # Option = 1  U0xy = H(phi(x,y))
    # Option = 2  U0xy = Image
    # UseLengthItem == 1 : |delta H(x)|
    # UseLengthItem == 2 : dirace(phi(x,y))|delta phi(x,y)|
    def LevelSetLoss(self,Image_, OutPut_FeatureMap, LabelMap, LevelSetLabel):
        self.Image_ = Image_
        self.OutPut_FeatureMap = OutPut_FeatureMap
        self.LabelMap = LabelMap

        # Transform both to float tensor
        FeatureMap = self.OutPut_FeatureMap.float()
        LabelMap = self.LabelMap.float()

        # This is the level set.
        # LevelSetFunction = self.OutputLevelSet(FeatureMap)
        LevelSetFunction = FeatureMap

        preNum = LevelSetFunction.size()[2] * LevelSetFunction.size()[3]
        HeavisideLevelSet =  self.HeavisideFunction(LevelSetFunction)

        # item 1 : |H(ls(x,y))-gt(x,y)|
        item1_2 = LabelMap  # 1*512*512
        item1_1 = torch.squeeze(HeavisideLevelSet, dim=1)  # 1*512*512

        ### 原始Loss计算方法
        # print(LabelMap)
        # a = classtensor.data.cpu().numpy()
        # print(classtensor)
        minums_ = item1_1 - item1_2
        # b = minums_.data.cpu().numpy()
        # print(minums_.size())
        item1_abs = torch.abs(minums_)
        item1_abs_pow = torch.pow(item1_abs, 2)
        Loss_item1 = torch.sum(item1_abs_pow) / preNum
        
        # # BCE loss
        # Loss_item1 = F.binary_cross_entropy(item1_1, item1_2)

        ## DICE loss
        # Loss_item1 = self.diceloss(item1_1, item1_2)

        print('Loss Item1=%f' % Loss_item1.data.cpu().numpy())

        # Item 2 the length of the zero-level set
        # UseLengthItem == 1 : |delta H(x)|
        # UseLengthItem == 2 : dirace(phi(x,y))|delta phi(x,y)|
        if self.UseHigh_Hfuntion == 0:
            HeavisideLevelSet_ = HeavisideLevelSet
        else:
            HeavisideLevelSet_ = self.HighElsHeavisideFunction(LevelSetFunction)

        if self.UseLengthItemType == 1:
            gradientX = self.Sobelx(HeavisideLevelSet_)
            gradientY = self.Sobely(HeavisideLevelSet_)
            gradientAll = torch.abs(gradientX) + torch.abs(gradientY)
            Loss_item2 = torch.sum(gradientAll) / preNum
            # print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())
        elif self.UseLengthItemType == 2:
            gradientX = self.Sobelx(LevelSetFunction)
            gradientY = self.Sobely(LevelSetFunction)
            gradientAll = torch.abs(gradientX) + torch.abs(gradientY)
            item2_part1 = self.DiracDeltaFunction(LevelSetFunction)
            # Multiply element-wise
            Loss_item2 = torch.sum(item2_part1 * gradientAll) / preNum
            # print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())
        elif self.UseLengthItemType == 3:
            gradientX = self.Sobelx(HeavisideLevelSet_)
            gradientY = self.Sobely(HeavisideLevelSet_)
            gradientAll = torch.abs(gradientX) + torch.abs(gradientY)
            gradientAll = gradientAll - 0.1
            gradientAll = self.relu_(gradientAll)
            a = gradientAll.data.cpu().numpy()
            b = (a != 0)
            c = b*1
            d = np.sum(c)
            #NonZeroNum = gradientAll.data.cpu().numpy()
            #NonZero = np.find(NonZeroNum>0)
            #NonZeroNum = np.sum()
            Loss_item2 = torch.sum(gradientAll) / d

        print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())

        # Item 3 the inner area and outer area
        # c_1 = torch.sum(SelectedTensor*SelectedTensor)/torch.sum(SelectedTensor)
        # c_2 = torch.sum(SelectedTensor*(1-SelectedTensor))/torch.sum(1-SelectedTensor)
        # Option = 1  U0xy = H(phi(x,y))
        # Option = 2  U0xy = Image
        # c_1, c_2 = GetC1_C2(Phi_t0=LevelSetFunction, Image_=InputImage, Option=option)
        if self.InnerAreaOption == 1:
            U0xy = HeavisideLevelSet
            c_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
            c_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)
        if self.InnerAreaOption == 2:
            U0xy = Image_
            c_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
            c_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)
        if self.InnerAreaOption == 3:
            U0xy = Image_
            c_1 = torch.sum(U0xy * LabelMap) / torch.sum(LabelMap)
            c_2 = torch.sum(U0xy * (1 - LabelMap)) / torch.sum(1 - LabelMap)

        item3_part1_1 = U0xy - c_1
        item3_part1_2 = torch.abs(item3_part1_1)
        item3_part1_3 = torch.pow(item3_part1_2, 2)
        item3_part1_4 = item3_part1_3 * HeavisideLevelSet
        item3_part1_loss = torch.sum(item3_part1_4)

        item3_part2_1 = U0xy - c_2
        item3_part2_2 = torch.abs(item3_part2_1)
        item3_part2_3 = torch.pow(item3_part2_2, 2)
        item3_part2_4 = item3_part2_3 * (1 - HeavisideLevelSet)
        item3_part2_loss = torch.sum(item3_part2_4)

        loss_item3 = (item3_part1_loss + item3_part2_loss) / preNum
        print('Loss Item3=%f' % loss_item3.data.cpu().numpy())

        # Item 4 the difference level set
        loss_Item4_1 = torch.pow(LevelSetFunction - LevelSetLabel, 2)
        loss_Item4 = torch.sum(loss_Item4_1) / preNum
        print('Loss Item4=%f' % loss_Item4.data.cpu().numpy())
        
        # Item 5 the shape prior
        gradientX = self.Sobelx(LevelSetFunction)
        gradientY = self.Sobely(LevelSetFunction)
        gradientAll = gradientX + gradientY
        Loss_item5_1 = torch.sum(gradientAll) / preNum
        loss_Item5 = self.relu_(Loss_item5_1*-1)
        
        print('Loss Item5=%f' % loss_Item5.data.cpu().numpy())
        
        #AllLoss = self.lambda_1 * Loss_item1 + self.lambda_2 * Loss_item2 + self.lambda_3 * loss_item3 + #self.Lamda_LevelSetDifference * loss_Item4
        
        # AllLoss = (1-self.lambda_1) * Loss_item1 + self.lambda_1 * loss_Item4 + self.lambda_2 * loss_Item5
        AllLoss = self.lambda_2 * loss_Item5 # 这里好像是后面做了消融实验？？？
        
        print('All loss = %f' % AllLoss.data.cpu().numpy())
        return AllLoss
