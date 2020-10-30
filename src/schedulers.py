from torch.optim.lr_scheduler import _LRScheduler


class ConstantLR(_LRScheduler):
    """constant learning rate schedulers

    Args:
        Optimizer (Object): Optimizer object
        last_epoch (int): Number of last epoch, used in resuming training
    Returns:
        Get schedulers
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

class PolynomialLR(_LRScheduler):
    """Polynomial learning rate schedulers

    Args:
        Optimizer (Object): Optimizer object
        max_iter (int): number of max iterations to be performed
        gamma (float): power in Polynomial learning rate decay equation
        last_epoch (int): Number of last epoch, used in resuming training
    Returns:
        Get schedulers
    """
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma

        print('>>> from schedulers.py --> class PolynomialLR --> __init__() <<<')
        print('>>>>>>>>>> [maxiter: {}] <<<<<<<<<<<<<<'.format(self.max_iter))

        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if float(self.max_iter) < self.last_epoch:
            raise ValueError('Number of training iterations exceeds that of the polynomial schedulers')
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


def get_scheduler(lr_policy): #*
    
    if lr_policy=="ConstantLR":
        scheduler_cls= ConstantLR
    
    elif lr_policy=="PolynomialLR":
        scheduler_cls= PolynomialLR

    return scheduler_cls