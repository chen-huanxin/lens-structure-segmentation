from configparser import ConfigParser

class Parameters():

    def __init__(self, path) -> None:
        self._path = path
        _conf = ConfigParser()
        _conf.read(self._path)

        # float type
        self.lambda_1 = float(_conf.get('Options', 'lambda_1'))
        self.lambda_2 = float(_conf.get('Options','lambda_2'))
        self.lambda_3 = float(_conf.get('Options','lambda_3'))
        self.e_ls = float(_conf.get('Options','e_ls'))
        self.lambda_shape = float(_conf.get('Options','lambda_shape'))
        self.lambda_CNN = float(_conf.get('Options','lambda_CNN'))
        self.Lamda_RNN = float(_conf.get('Options','Lamda_RNN'))
        self.lr = float(_conf.get('Options','lr'))
        self.lr_decay_epoch = float(_conf.get('Options','lr_decay_epoch'))
        self.Highe_ls = float(_conf.get('Options','Highe_ls'))
        self.Lamda_LevelSetDifference = float(_conf.get('Options','Lamda_LevelSetDifference'))

        # int type
        self.option_ = int(_conf.get('Options','option_'))
        self.n_class = int(_conf.get('Options','n_class'))
        self.InnerAreaOption = int(_conf.get('Options','InnerAreaOption'))
        self.UseLengthItemType = int(_conf.get('Options','UseLengthItemType'))
        self.UseHigh_Hfuntion = int(_conf.get('Options','UseHigh_Hfuntion'))
        self.isShownVisdom = int(_conf.get('Options','isShownVisdom'))
        self.GRU_Number = int(_conf.get('Options','GRU_Number'))
        self.RNNEvolution = int(_conf.get('Options','RNNEvolution'))
        self.ShapePrior = int(_conf.get('Options','ShapePrior'))
        self.inputSize = int(_conf.get('Options','inputSize'))
        self.gpu_num = int(_conf.get('Options','gpu_num'))
        self.CNNEvolution = int(_conf.get('Options','CNNEvolution'))
        self.GRU_Dimention = int(_conf.get('Options','GRU_Dimention'))
        self.n_epochs = int(_conf.get('Options','n_epochs'))
        self.lr_decay = int(_conf.get('Options','lr_decay'))
        self.batch_size = int(_conf.get('Options','batch_size'))
        self.img_size = int(_conf.get('Options','img_size'))
        self.random_seed = int(_conf.get('Options','random_seed'))
        self.UseHigh_Hfuntion = int(_conf.get('Options','UseHigh_Hfuntion'))
        self.NormalizationOption = int(_conf.get('Options','NormalizationOption'))
        self.NormalizationOption_3_Alpha = int(_conf.get('Options','NormalizationOption_3_Alpha'))
        self.NormalizationOption_3_Beta = int(_conf.get('Options','NormalizationOption_3_Beta'))
        self.UseSigmoid = int(_conf.get('Options','UseSigmoid'))
        self.UseTanh = int(_conf.get('Options','UseTanh'))
        self.IfFineTune = int(_conf.get('Options','IfFineTune'))

        # str type
        self.LevelSetLabelDir = _conf.get('Options','LevelSetLabelDir')
        self.SaveDir = _conf.get('Options','SaveDir')
        self.LogDir = _conf.get('Options','LogDir')
        self.ModelName = _conf.get('Options','ModelName')
        self.DataSetDir = _conf.get('Options','DataSetDir')
        self.FineTuneModelDIr = _conf.get('Options','FineTuneModelDIr')


    def getDict(self):
        Options={
            'InnerAreaOption': self.InnerAreaOption,
            'UseLengthItemType': self.UseLengthItemType,
            'UseHigh_Hfuntion': self.UseHigh_Hfuntion,
            'isShownVisdom': self.isShownVisdom,
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_3': self.lambda_3,
            'lambda_shape': self.lambda_shape,
            'lambda_CNN': self.lambda_CNN,
            'Lamda_RNN': self.Lamda_RNN,
            'GRU_Number':0,
            'RNNEvolution': self.RNNEvolution,
            'ShapePrior': self.ShapePrior,
            'inputSize':( self.img_size, self.img_size),
            'ShapeTemplateName': '/home/intern1/guanghuixu/resnet/shapePrior/1390_L_004_110345.pkl',
            'gpu_num': self.gpu_num,
            'CNNEvolution': self.CNNEvolution,
            'GRU_Dimention': self.GRU_Dimention,
            'e_ls': self.e_ls,
            'option_': self.option_,
            'UseHigh_Hfuntion': self.UseHigh_Hfuntion,
            'Highe_ls': self.Highe_ls,   # 1/1024
            'EpochNum': self.n_epochs,
            'DownEpoch': self.lr_decay_epoch,
            'LearningRate': self.lr,
            'LevelSetLabelDir': self.LevelSetLabelDir,
            'DataSetDir': self.DataSetDir,
            'SaveDir': self.SaveDir,
            'LogDir': self.LogDir,
            'ModelName': self.ModelName,
            'NormalizationOption': self.NormalizationOption,
            'NormalizationOption_3_Alpha': self.NormalizationOption_3_Alpha,
            'NormalizationOption_3_Beta': self.NormalizationOption_3_Beta,
            'Lamda_LevelSetDifference': self.Lamda_LevelSetDifference,
            'UseSigmoid': self.UseSigmoid,
            'UseTanh': self.UseTanh,
            'IfFineTune': self.IfFineTune,
            'FineTuneModelDIr': self.FineTuneModelDIr
        }
        return Options


if __name__ == '__main__':
    params = Parameters('config.ini')
    print(params.lambda_1)
    print(params.LevelSetLabelDir)
    # tmp = props(params)
    pass