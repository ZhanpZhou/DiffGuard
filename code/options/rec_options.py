class RecOptions(object):
    @staticmethod
    def modify_commandline_options(parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='Rec', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/reconstruction', help='models are saved here')
        parser.add_argument('--load_dir', type=str, default=None, help='model paths to be loaded')
        parser.add_argument('--vis_dir', type=str, default='./reconstruction_vis', help='model outputs are saved here')
        parser.add_argument('--remark', type=str, default='', help='special remarks, appended to visualize dir')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='ctslice_diffusion', help='chooses train dataset')
        parser.add_argument('--dataset', type=object, default=None, help='created dataset')
        parser.add_argument('--dataset_id', type=int, default=0, help='index of train dataset')
        parser.add_argument('--v_dataset_mode', type=str, default='ctslice_diffusion', help='chooses valid dataset')
        parser.add_argument('--v_dataset', type=object, default=None, help='created v_dataset')
        parser.add_argument('--v_dataset_id', type=int, default=0, help='index of valid and test dataset')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--v_batch_size', type=int, default=32, help='valid input batch size')
        parser.add_argument('--drop_last', type=int, default=0)
        parser.add_argument('--serial_batches',type=bool, default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')

        parser.add_argument('--train_datasets', type=list, default=[[]])
        parser.add_argument('--valid_datasets', type=list, default=[[]])
        parser.add_argument('--test_datasets', type=list, default=[[]])

        parser.add_argument('--classes', type=str, default=','.join(['normal', 'thymoma', 'benign cysts', 'germ cell tumor', 'neurogenic tumor', 'thymic carcinoma']))
        parser.add_argument('--modality', type=str, default='CE', choices=('CE', 'PL'))
        parser.add_argument('--direction', type=str, default='axial')

        parser.add_argument('--input_type', type=str, default='image', choices=('image', 'mask'))
        parser.add_argument('--sample_wwl', type=int, default=0)
        parser.add_argument('--data_norm_type', type=str, default='gray', choices=('imagenet', 'mmdetection', 'gray', 'normal', 'activitynet', 'kinetics', 'competition'))
        parser.add_argument('--ww_wl_list', default=[[400, 0]])
        parser.add_argument('--input_h', type=int, default=256)
        parser.add_argument('--input_w', type=int, default=256)
        parser.add_argument('--condition', type=str, default='saliency_tumor', choices=('tumor', 'saliency', 'saliency_tumor', 'none'))
        parser.add_argument('--sample_strategy', type=str, default='original')
        parser.add_argument('--positive_only', type=int, default=0)
        parser.add_argument('--test_samples', default=[])
        
        parser.set_defaults(model='gen', method_name='gen', lr=1e-4, save_mode='epoch', max_epoch=200, save_epoch_freq=40)

        return parser

    def __init__(self):
        pass