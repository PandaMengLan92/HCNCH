#!coding:utf-8
import os
import numpy as np
import random
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.ramps import WarmupCosineLrScheduler
from utils import datasets
from utils.ramps import exp_warmup
from utils.config import parse_commandline_args
from utils.data_utils import DataSetWarpper
from utils.data_utils import TwoStreamBatchSampler
from utils.data_utils import TransformTwice as twice
from convlarge import CNN

from MeanTeacher import Trainer


def create_loaders(labelset, unlabelset, queryset, database, config):

    #read index of loader
    #queryset = DataSetWarpper(queryset, config.num_classes)
    #database = DataSetWarpper(database, config.num_classes)

    labelset = DataSetWarpper(labelset, config.num_classes)
    ## supervisd batch loader

    label_loader = torch.utils.data.DataLoader(labelset,
                                               batch_size=config.sup_batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               drop_last=True)
    ## unsupervised batch loader

    unlab_loader = torch.utils.data.DataLoader(unlabelset,
                                               batch_size=config.usp_batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               drop_last=True)
    ## query batch loader
    query_loader = torch.utils.data.DataLoader(queryset,
                                               batch_size=config.sup_batch_size,
                                               shuffle=False,
                                               num_workers=2 * config.workers,
                                               pin_memory=True,
                                               drop_last=False)

    ## database batch loader
    database_loader = torch.utils.data.DataLoader(database,
                                                  batch_size=config.sup_batch_size,
                                                  shuffle=False,
                                                  num_workers=2 * config.workers,
                                                  pin_memory=True,
                                                  drop_last=False)

    return label_loader, unlab_loader, query_loader, database_loader


def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'cos-warmup':
        scheduler = WarmupCosineLrScheduler(optimizer, T_warmup=config.rampup_length,
                                            T_max=200, eta_min=config.min_lr)

    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def run(config):
    print(config)
    print("pytorch version : {}".format(torch.__version__))

    #set seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.random:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    ## create save directory
    if config.save_freq!=0 and not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    ## prepare data
    dconfig = datasets.load[config.dataset](config.data_root, config.num_classes, config.num_labels, config.num_unlabel, config.num_query)

    loaders = create_loaders(**dconfig, config=config)


    ## prepare architecture
    net = CNN(config.num_classes, config.drop_ratio, config.code_bits, True)
    net = net.to(device)
    optimizer = create_optim(net.parameters(), config)
    scheduler = create_lr_scheduler(optimizer, config)

    ## run the model

    net2 = CNN(config.num_classes, config.drop_ratio, config.code_bits, True)
    net2 = net2.to(device)
    trainer = Trainer(net, net2, optimizer, device, config)

    trainer.loop(config.epochs, *loaders, scheduler=scheduler)


if __name__ == '__main__':
    config = parse_commandline_args()
    #for i in [2.0, 2.5, 3.0, 3.5]:
    config.cons_weight = 1
    config.contras_weight = 0.1
    config.consis_t = 2.0
    config.contras_t1 = 2.5
    config.contras_t2 = 2.5
    
    print('consis_t:', config.consis_t, 'contras_t1:', config.contras_t1, 'contras_t2', config.contras_t2)
    run(config)

    """
    j = 0.1
    for i in [1.0]:
        config.cons_weight = i
        config.contras_weight = j
        print('con_weight:', i, 'contras_weight:', j)
        run(config)
    """
