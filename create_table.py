import os
import configparser
import xlwt

class TableMaker():

    def __init__(self, ini_dir, log_root):
        self._ini_dir = ini_dir
        self._log_root = log_root

        self._ini_list = os.listdir(self._ini_dir)
        
    def readConf(self, path):
        conf = configparser.ConfigParser()
        conf.read(path)

        # float type=1
        lambda_1 = conf.get('Options', 'lambda_1')
        lambda_2 = conf.get('Options', 'lambda_2')
        lambda_3 = conf.get('Options', 'lambda_3')
        e_ls = conf.get('Options', 'e_ls')
        lr = conf.get('Options', 'lr')
        lr_decay_epoch = conf.get('Options', 'lr_decay_epoch')
        Highe_ls = conf.get('Options', 'Highe_ls')

        # int type=2
        option_ = int(conf.get('Options', 'option_'))
        InnerAreaOption = int(conf.get('Options', 'InnerAreaOption'))
        UseLengthItemType = int(conf.get('Options', 'UseLengthItemType'))
        UseHigh_Hfuntion = int(conf.get('Options', 'UseHigh_Hfuntion'))
        CNNEvolution = int(conf.get('Options', 'CNNEvolution'))
        n_epochs = int(conf.get('Options', 'n_epochs'))
        UseHigh_Hfuntion = int(conf.get('Options', 'UseHigh_Hfuntion'))
        NormalizationOption = int(conf.get('Options', 'NormalizationOption'))
        NormalizationOption_3_Alpha = int(conf.get('Options', 'NormalizationOption_3_Alpha'))
        NormalizationOption_3_Beta = int(conf.get('Options', 'NormalizationOption_3_Beta'))
        UseSigmoid = int(conf.get('Options', 'UseSigmoid'))
        UseTanh = int(conf.get('Options', 'UseTanh'))
        IfFineTune = int(conf.get('Options', 'IfFineTune'))
        Lamda_LevelSetDifference = conf.get('Options', 'Lamda_LevelSetDifference')

        # str type = 3
        ModelName = conf.get('Options', 'ModelName')
        FineTuneModelDIr = conf.get('Options', 'FineTuneModelDIr')

        options = {
            'InnerAreaOption': InnerAreaOption,
            'UseLengthItemType': UseLengthItemType,
            'UseHigh_Hfuntion': UseHigh_Hfuntion,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'CNNEvolution': CNNEvolution,
            'e_ls': e_ls,
            'option_': option_,
            'Highe_ls': Highe_ls,  # 1/1024
            'EpochNum': n_epochs,
            'DownEpoch': lr_decay_epoch,
            'LearningRate': lr,
            'ModelName': ModelName,
            'NormalizationOption': NormalizationOption,
            'NormalizationOption_3_Alpha': NormalizationOption_3_Alpha,
            'NormalizationOption_3_Beta': NormalizationOption_3_Beta,
            'Lamda_LevelSetDifference': Lamda_LevelSetDifference,
            'UseSigmoid': UseSigmoid,
            'UseTanh': UseTanh,
            'IfFineTune': IfFineTune,
            'FineTuneModelDIr': FineTuneModelDIr
        }

        return options

    def generateTr(self, name, score):
        return f'<tr><td>{name}</td><td>{score}</td></tr>'

    def writeExcel(self, save_name: str='summary.xls'):
        excel_fh = xlwt.Workbook()
        table = excel_fh.add_sheet('info')

        cnt = 0
        row = 0
        first_flag = True

        for ini_name in self._ini_list:
            cnt += 1
            print(f'processing {cnt}/{len(self._ini_list)}')
            row += 1
            ini_file = os.path.join(self._ini_dir, ini_name)
            configs = self.readConf(ini_file)
            sub_dir = configs['ModelName']
            log_file = os.path.join(self._log_root, sub_dir, 'test_log.txt')

            with open(log_file, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                items = last_line.split('\t')
                for item in items:
                    key_val = item.split(':')
                    configs[key_val[0]] = key_val[1]

            all_keys = list(configs.keys())
            if first_flag:
                first_flag = False
                for idx, key in enumerate(all_keys):
                    table.write(0, idx, key)

            for name, score in configs.items():
                idx = all_keys.index(name)
                table.write(row, idx, score)

        excel_fh.save(save_name)

    def writeHtml(self, save_name: str='summary.html'):

        with open(save_name, 'w') as f_html:
            for ini_name in self._ini_list:
                ini_file = os.path.join(self._ini_dir, ini_name)
                configs = self.readConf(ini_file)
                sub_dir = configs['ModelName']
                log_file = os.path.join(self._log_root, sub_dir, 'test_log.txt')

                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    last_line = lines[-1]
                    items = last_line.split('\t')
                    for item in items:
                        key_val = item.split(':')
                        configs[key_val[0]] = key_val[1]

                tds = [self.generateTr(name, score) for name, score in configs.items()]
                f_html.writelines('<table border=1>')
                f_html.writelines('<tr><th>Name</th><th>Score</th></tr>')
                f_html.writelines('\n' + tds)
                f_html.writelines('</table>')


if __name__ == '__main__':
    tm = TableMaker(os.path.join(os.getcwd(), 'scripts', 'hyperparameter'), '/home/chenhuanxin/logs')
    