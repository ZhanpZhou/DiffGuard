import argparse
import json
import os
import models
import options
import datasets
import schedulers
from util import basic

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, task=None, **kwargs):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.parser = self.gather_options(task, **kwargs)
        self.parser.set_defaults(**kwargs)
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # for reproducibility
        parser.add_argument('--seed', type=int, default=0, help='seed of all random number generators')

        # task
        parser.add_argument('--task', type=str, default='segmentation', help='name of the experiment. It decides which task option to use')
        parser.add_argument('--model', type=str, default='seg', help='chooses which model to use.')
        parser.add_argument('--l_state', type=str,default='train', help='learning state')
        parser.add_argument('--greater_is_better', type=int, default=1)
        
        # data parallel and federated learning
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--parallel_mode', type=str, default='dataparallel', choices=('dataparallel', 'distributed', 'federated'))
        parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--world_size', type=int, default=1)

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--plot_info', default='right,wrong', type=str, help='plot correct or wrong images')
        parser.add_argument('--visualize', type=int, default=0)
        parser.add_argument('--print_freq', type=int, default=50,
                            help='frequency of showing training results on console')
        parser.add_argument('--v_print_freq', type=int, default=50,
                            help='frequency of showing validation results on console')
        parser.add_argument('--save_stat_info', type=int, default=1)

        self.initialized = True
        return parser

    def gather_options(self, task, **kwargs):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,conflict_handler="resolve")
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        for key in ('task', 'model'):
            if key in kwargs:
                parser.set_defaults(**{key: kwargs[key]})

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify task-related
        if task is not None:
            opt.task = task
        task_option_setter = options.get_option_setter(opt.task)
        parser = task_option_setter(parser)

        # modify model-related parser options
        opt, _ = parser.parse_known_args()
        model_option_setter = models.get_option_setter(opt.model)
        parser = model_option_setter(parser)

        # parse again with new defaults
        opt, _ = parser.parse_known_args()

        for key in ('dataset_mode', 'v_dataset_mode'):
            if key in kwargs:
                parser.set_defaults({key: kwargs[key]})

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        if opt.dataset_mode != opt.v_dataset_mode:
            dataset_name = opt.v_dataset_mode
            dataset_option_setter = datasets.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser)

        # modify scheduler-related parser options
        
        # save and return the parser
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        basic.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def opt_revise(self,opt):
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')

        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
                
        str_info = opt.plot_info.split(',')
        opt.plot_info = []
        for info in str_info:
            opt.plot_info.append(info)

        # class divisions
        if hasattr(opt, 'classes'):
            opt.classes_ = opt.classes
            classes = opt.classes.split('-')
            opt.classes = [s.split(',') for s in classes]

        # sample weights
        if hasattr(opt, 'sample_weights'):
            opt.sample_weights = [float(x) for x in opt.sample_weights.split(',')]
        if hasattr(opt, 'resample_ratios'):
            opt.resample_ratios = [float(x) for x in opt.resample_ratios.split(',')]

        # backbone
        if hasattr(opt, 'backbones'):
            opt.backbones_ = opt.backbones
            opt.backbones = opt.backbones.split(',')
        if hasattr(opt, 'archs'):
            opt.archs_ = opt.archs
            opt.archs = opt.archs.split(',')

        if hasattr(opt, 'in_channels'):
            opt.in_channels = [int(x) for x in opt.in_channels.split(',')]
        if hasattr(opt, 'out_channels'):
            opt.out_channels = [int(x) for x in opt.out_channels.split(',')]

        # classification heads
        if hasattr(opt, 'heads'):
            heads = opt.heads.split('-')
            opt.heads = []
            for i, head_config in enumerate(heads):
                splits = head_config.split('_')
                backbones = [int(x) for x in splits[0].split(',')]
                class_num = int(splits[1])
                opt.heads.append((backbones, class_num))
        if hasattr(opt, 'labels'):
            opt.labels_ = opt.labels
            opt.labels = opt.labels.split(',')

        # recall thresholds
        if hasattr(opt, 'recall_thresholds') and opt.recall_thresholds is not None:
            opt.recall_thresholds = json.loads(opt.recall_thresholds)

        # loss weights
        if hasattr(opt, 'loss_weights'):
            opt.loss_weights = [float(x) for x in opt.loss_weights.split(',')]

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.parser.parse_args()
        self.opt = self.opt_revise(opt)
        return self.opt

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # network saving and loading parameters
        parser.add_argument('--start_epoch', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--max_epoch', type=int, default=4800, help='maximum epoch number')
        parser.add_argument('--max_iter', type=int, default=5000000, help='maximum epoch number')

        parser.add_argument('--valid_model', type=int, default=0, help='valid the model')
        parser.add_argument('--valid_metric', type=str, default='instance_auc')
        parser.add_argument('--valid_mode', type=str, default='iter', choices=('epoch', 'iter'))
        parser.add_argument('--valid_epoch_freq', type=int, default=400, help='frequency of validating the latest model')
        parser.add_argument('--valid_iter_freq', type=int, default=400, help='frequency of validating the latest model')
        parser.add_argument('--save_mode', type=str, default='iter', choices=('epoch', 'iter'))
        parser.add_argument('--save_epoch_freq', type=int, default=400, help='frequency of saving checkpoints')
        parser.add_argument('--save_iter_freq', type=int, default=400, help='frequency of saving checkpoints')
        parser.add_argument('--scheduler_metric', type=str, default='instance_auc')
        
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--test_model', type=int, default=0, help='test the model')

        # optimizer parameters
        parser.add_argument('--optim', type=str, default='AdamW', help='# the name of optimizer')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--nesterov', type=bool, default=True, help='nesterov term of SGD')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum term of SGD')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 term of Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 term of Adam')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        parser.add_argument('--grad_clip_value', type=float, default=1, help='grad clip value')
        parser.add_argument('--grad_iter_size', type=int, default=1, help='# grad iter size')

        # scheduler
        parser.add_argument('--lr_policy', type=str, default='mstep', choices=('linear','step','plateau','cosine','cosineWR','mstep'))
        parser.add_argument('--milestones', type=list, default=[], help='mstep scheduler')
        parser.add_argument('--gamma', type=float, default=0.3)
        parser.add_argument('--warmup', type=int, default=0)
        parser.add_argument('--warm_epoch', type=int, default=3)
        parser.add_argument('--lr_wait_epoch', type=int, default=4, help='plateau scheduler')
        parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='plateau scheduler')
        parser.add_argument('--lr_decay_iters', type=int, default=3, help='step scheduler, multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--patient_epoch', type=int, default=10000, help='')
        parser.add_argument('--T_0', type=int, default=5, help='T_max in CosineAnnealingLR and T_0 in CosineAnnealingWarmRestarts')
        parser.add_argument('--T_mult', type=int, default=1, help='T_mult in CosineAnnealingWarmRestarts')
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def gather_options(self, task, **kwargs):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = BaseOptions.gather_options(self, task, **kwargs)

        # get the basic options

        opt, _ = parser.parse_known_args()

        # modify sheduler-related parser options
        policy_name = opt.lr_policy
        if policy_name not in schedulers.basic_schedulers:
            policy_option_setter = schedulers.get_option_setter(policy_name)
            parser = policy_option_setter(parser)
        return parser


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        return parser

    def gather_options(self, task, **kwargs):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = BaseOptions.gather_options(self, task, **kwargs)
        return parser

