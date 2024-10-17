from .base_scheduler import BaseScheduler

class MstepScheduler(BaseScheduler):
    def __init__(self, optimizer, opt):
        super(MstepScheduler, self).__init__(optimizer,opt)
        self.optimizer = optimizer
        self.best_value = None

        if self.opt.warmup:
            self.lr = self.warm_start_lr
        else:
            self.lr = self.opt.lr

    def step(self,value, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.opt.warmup and epoch < self.opt.warm_epoch:
            self.lr = self.warm_up(epoch)

        if self.best_value == None or value > self.best_value:
            self.best_value = value
            self.wait_epoch = 0
        else:
            self.wait_epoch += 1
            if self.wait_epoch > self.opt.lr_wait_epoch:
                self.wait_epoch = 0
                self.lr = self.lr * self.opt.lr_decay_factor

    def get_last_lr(self):
        return [self.lr for param_group in self.optimizer.param_groups]

    @staticmethod
    def modify_commandline_options(parser):
        return parser
