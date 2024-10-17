import contextlib
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch

from util.basic import *
from util.metrics import TimeMeter
from models import create_model
from datasets import create_dataset
from util.visualizer import Visualizer
from options.base_options import TrainOptions


def train(opt):
    random.seed(opt.seed + opt.rank)
    np.random.seed(opt.seed + opt.rank)
    torch.manual_seed(opt.seed + opt.rank)
    torch.cuda.manual_seed_all(opt.seed + opt.rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    dataset = create_dataset(opt, 'train')  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)

    if opt.valid_model:
        v_dataset = create_dataset(opt, 'valid')
        print('The number of validation samples = %d' % len(v_dataset))

    if opt.test_model:
        t_datasets = []
        for i, datasets in enumerate(opt.test_datasets):
            opt.v_dataset_id = i
            t_dataset = create_dataset(opt, 'test')
            print('The number of test samples = %d' % len(t_dataset))
            t_datasets.append(t_dataset)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.train_dataset = dataset.dataset
    if opt.valid_model:
        model.valid_dataset = v_dataset.dataset

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    valid_iters = 0
    stop = False

    iter_time_meter = TimeMeter()
    data_time_meter = TimeMeter()
    epoch_time_meter = TimeMeter()

    print('Start to train')

    for epoch in range(opt.start_epoch, opt.max_epoch + 1):    # outer loop for different epochs; we save the model by <start_epoch>, <start_epoch>+<save_latest_freq>
        if opt.parallel_mode == 'distributed':
            dataset.sampler.set_epoch(epoch)
        dataset.dataset.prepare_new_epoch()

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_time_meter.start()  # timer for entire epoch
        data_time_meter.start()
        iter_time_meter.start()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record()
            iter_time_meter.start()
            visualizer.reset()
            total_iters += 1
            epoch_iter += 1

            # stage 1: backpropagation and update parameters
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            iter_time_meter.record()

            if total_iters % opt.print_freq == 0 and opt.rank == 0:    # print training losses and save logging information to the disk
                visualizer.print_current_info(epoch, total_iters, model, iter_time_meter.val, data_time_meter.val)
                # visualizer.plot_current_info(model, total_iters, ptype='train')

            iter_time_meter.reset()
            data_time_meter.reset()

            # stage 2: validation and update learning rate
            if opt.valid_model and opt.valid_mode == 'iter':
                valid_model = True if total_iters % opt.valid_iter_freq == 0 and opt.rank == 0 else False             
                if valid_model:
                    model.reset_meters()
                    model.clear_info()
                    with torch.no_grad():
                        model.validation(v_dataset, visualizer, valid_iters, epoch)
                    model.save_optimal_networks(visualizer)
                    model.reset_meters()
                    valid_iters += 1

            model.update_learning_rate()

            if model.wait_epoch > opt.patient_epoch:
                stop = True
                print('early stop at %d / %d' % (epoch, epoch_iter))
                break

            data_time_meter.start()
            iter_time_meter.start()

            # stage 3: save state dict
            if opt.save_mode == 'iter':
                if total_iters % opt.save_iter_freq == 0 and opt.rank == 0 :
                    model.save_networks(f'iter{total_iters}')
                    # model.save_optimizer_scheduler(f'iter{total_iters}')

            if total_iters == opt.max_iter:
                stop = True
                print('finish at iter {}'.format(total_iters))
                break
        
        if opt.save_mode == 'epoch':
            if epoch % opt.save_epoch_freq == 0 and opt.rank == 0 :
                model.save_networks(f'epoch{epoch}')
                # model.save_optimizer_scheduler(f'epoch{epoch}')

        model.update_metrics('global')
        if opt.rank == 0:
            visualizer.print_global_info(epoch, epoch_iter, model, iter_time_meter.sum/60,data_time_meter.sum/60)
            # visualizer.plot_global_info(model, valid_iters, ptype='train')
        model.reset_meters()
        model.clear_info()

        if opt.valid_model and opt.valid_mode == 'epoch':
            valid_model = True if epoch % opt.valid_epoch_freq == 0 and opt.rank == 0 else False             
            if valid_model:
                with torch.no_grad():
                    model.validation(v_dataset, visualizer, valid_iters, epoch)
                model.save_optimal_networks(visualizer)
                model.reset_meters()
                valid_iters += 1

        if stop:
            break

        epoch_time_meter.record()
        epoch_time_meter.start()
        
        model.next_epoch()

        print('End of epoch %d, iter %d \t Time Taken: %d hours' % (epoch, total_iters, epoch_time_meter.sum/3600.))

    if not hasattr(model, 'best_epoch_metrics'):
        model.best_epoch_metrics = 'No best epoch metric.'

    if opt.rank == 0:
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % model.best_epoch_metrics)  # save the message

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.rank == 0:
        train(opt)
    else: 
        with contextlib.redirect_stdout(None):
            train(opt)