import configparser
import os
import stat
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build ini file')

    parser.add_argument(
        '--root',
        '-r',
        default = os.getcwd(),
        help='project root path',
    )

    parser.add_argument(
        '--dataset',
        '-d',
        default = 'nucleus',
        choices = ['nucleus', 'cortex'],
        help='dataset name',
    )

    parser.add_argument(
        '--subset',
        '-s',
        default='CV1',
        choices=['CV1', 'CV2'],
        help='sub-dataset name',
    )

    parser.add_argument(
        '--activation',
        '-a',
        default='sigmoid',
        choices=['sigmoid', 'tanh'],
        help='sub-dataset name',
    )

    parser.add_argument(
        '--gpu',
        '-g',
        type=int,
        default=0,
        help='use gpu#',
    )

    parser.add_argument(
        '--lr',
        '-l',
        type=float,
        default=0.0005,
        help='set learning rate',
    )

    parser.add_argument(
            '--method',
            '-m',
            type=str,
            default='Proposed',
            choices=['Proposed', 'Baseline'],
            help='use method',
    )

    parser.add_argument("--n_epochs", '-n', default=60, type=int, help='epoch num')

    args = parser.parse_args()

    return args

def build_ini(args, model_name):
    conf = configparser.ConfigParser()

    try:
        conf.add_section('Options')
    except:
        print('the key is exits')

    conf.set('Options', 'InnerAreaOption', '1')
    conf.set('Options', 'UseLengthItemType', '1')
    conf.set('Options', 'UseHigh_Hfuntion', '0')
    if args.dataset == 'nucleus':
        conf.set('Options', 'lambda_1', '0.5')
        conf.set('Options', 'lambda_2', '0.5')
    else:
        conf.set('Options', 'lambda_1', '0.7')
        conf.set('Options', 'lambda_2', '0.1')
    conf.set('Options', 'lambda_3', '0.0')
    conf.set('Options', 'lambda_shape', '0.0')
    conf.set('Options', 'lambda_CNN', '1.0')
    conf.set('Options', 'Lamda_RNN', '0.0')
    conf.set('Options', 'GRU_Number', '1')
    conf.set('Options', 'RNNEvolution', '0')
    conf.set('Options', 'ShapePrior', '0')
    conf.set('Options', 'inputSize', '512')
    conf.set('Options', 'gpu_num', str(args.gpu))
    conf.set('Options', 'CNNEvolution', '1')
    conf.set('Options', 'GRU_Dimention', '2')
    conf.set('Options', 'e_ls', str(1/32.0))
    conf.set('Options', 'option_', '2')
    conf.set('Options', 'Highe_ls', str(1/1024.0))
    conf.set('Options', 'UseHigh_Hfuntion', '1')
    conf.set('Options', 'lr', str(args.lr))
    conf.set('Options', 'n_epochs', str(args.n_epochs))
    conf.set('Options', 'lr_decay', '10')
    conf.set('Options', 'lr_decay_epoch', '10')
    conf.set('Options', 'batch_size', '1')
    conf.set('Options', 'img_size', '512')
    conf.set('Options', 'n_class', '2')
    conf.set('Options', 'random_seed', '2321')
    conf.set('Options', 'LevelSetLabelDir', os.path.join(os.path.expanduser('~'), 'datasets/lens-segmentation', args.dataset, args.subset, 'level_set_label'))
    conf.set('Options', 'DatasetDir', os.path.join(os.path.expanduser('~'), 'datasets/lens-segmentation', args.dataset, args.subset))
    conf.set('Options','LogDir', os.path.join(os.path.expanduser('~'), 'logs'))
    conf.set('Options','NormalizationOption','3')
    conf.set('Options','NormalizationOption_3_Alpha', '10')
    conf.set('Options','NormalizationOption_3_Beta', '10')
    conf.set('Options', 'Lamda_LevelSetDifference', '2')
    conf.set('Options', 'ModelName', model_name)
    is_sigmoid = int(args.activation == 'sigmoid')
    conf.set('Options', 'UseSigmoid', str(is_sigmoid))
    conf.set('Options', 'UseTanh', str(1 - is_sigmoid))
    conf.set('Options', 'IfFineTune', '0')

    conf.set('Options', 'isShownVisdom', '0')
    conf.set('Options','SaveDir', os.path.join(args.root, 'models/'))
    conf.set('Options', 'data_path', os.path.join(os.path.expanduser('~'), 'datasets/lens-segmentation', args.dataset, args.subset, 'level_set_label'))
    conf.set('Options', 'ShapeTemplateName', '/home/intern1/guanghuixu/resnet/shapePrior/1390_L_004_110345.pkl')
    conf.set('Options', 'results', os.path.join(args.root, 'results'))
    conf.set('Options', 'FineTuneModelDir', os.path.join(args.root, 'models/'))

    return conf

if __name__ == '__main__':
    args = parse_args()

    root = args.root # the path of project
    python_file_path = os.path.join(root, 'main.py')

    #GetModelName
    cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = cur_time + '_' + args.dataset + '_' + args.subset + '_' + args.activation
    save_ini_dir = os.path.join(root, 'scripts/hyperparameter')
    save_bash_dir = os.path.join(root, 'scripts/bash')
    save_ini_name = os.path.join(save_ini_dir, model_name + '.ini')
    save_bash_name = os.path.join(save_bash_dir, model_name + '.sh')

    if not os.path.exists('scripts'):
        os.mkdir('scripts')

    if not os.path.exists(save_ini_dir):
        os.mkdir(save_ini_dir)

    if not os.path.exists(save_bash_dir):
        os.mkdir(save_bash_dir)

    #Build INI
    conf = build_ini(args, model_name)
    with open(save_ini_name, 'w') as f_ini:
        conf.write(f_ini)

    #Build Bash
    with open(save_bash_name, 'w') as f_bash:
        f_bash.write(f'py_file={python_file_path}\n')
        f_bash.write(f'ininame={save_ini_name}\n')
        f_bash.write(f'python $py_file --ini_path=$ininame --model={args.method}\n')

    os.chmod(save_bash_name, stat.S_IRWXU)
    if os.path.exists('run.sh'):
        os.unlink('run.sh')
    os.symlink(save_bash_name, 'run.sh')


