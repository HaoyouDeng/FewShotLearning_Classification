import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

from methods import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

from loguru import logger

if __name__ == '__main__':
    params = parse_args('test')

    GPU = params.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    torch.cuda.set_device(GPU)
    CUDA = 'cuda:' + str(GPU)

    params.checkpoint_dir = '%s/%s'%(params.save_dir, params.name)
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
    logger.add('{}/logs/test/{}.log'.format(params.checkpoint_dir, start_time))

    acc_all = []
    iter_num = 2000
    image_size = 224

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    # define dataloader for test
    logger.info('Loading target dataset: {}!'.format(params.testset))
    novel_file = os.path.join(params.data_dir, params.testset, 'novel.json')
    datamgr = SetDataManager(image_size, n_query=15, n_eposide=iter_num, **few_shot_params)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)

    # define model
    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6': 
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S': 
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    # load model checkpoint
    if params.test_file is not None:
        checkpoint_dir = params.test_file
    else:
        checkpoint_dir = '%s/%s/best_model.tar' % (params.save_dir, params.name)

    tmp = torch.load(checkpoint_dir, map_location=CUDA)
    model.load_state_dict(tmp['state'])

    logger.info('Load checkpoint: {}'.format(checkpoint_dir))
    logger.info('Epoch: {}'.format(tmp['epoch']))

    logger.info('Start testing...')

    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

    else:
        if params.adaptation:
            acc_mean, acc_std = model.test_loop( novel_loader, adaptation=True)
        else:
            acc_mean, acc_std = model.test_loop( novel_loader)
    
    logger.success('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
