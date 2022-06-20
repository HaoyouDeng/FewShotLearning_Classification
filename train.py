from nbformat import validate
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import json
from tqdm import tqdm

from methods import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file 

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    max_epoch = 0
    tqdm_gen = tqdm(range(start_epoch, stop_epoch), total=(stop_epoch-start_epoch), ncols=100)
    for epoch in tqdm_gen:
        # train
        model.train()
        epoch_loss = model.train_loop(epoch, base_loader,  optimizer)
        tb_writer.add_scalar('train_loss', epoch_loss, epoch)

        # val
        model.eval()
        acc_mean, acc_std = model.test_loop(val_loader)
        tb_writer.add_scalar('val_acc', acc_mean, epoch)
        tqdm_gen.set_description('%d Test Acc = %4.2f%% +- %4.2f%%' %(len(val_loader),  acc_mean, 1.96* acc_std/np.sqrt(len(val_loader))))

        if acc_mean > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            max_acc = acc_mean
            max_epoch = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    
    logger.info("Best accuracy {:f}, Best epoch:{}".format(max_acc, max_epoch))

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    # # For debug
    # params.model = 'Conv4'
    # params.method = 'baseline'
    # params.fix_layers = 1
    # params.stop_epoch = 100
    # params.gpu = 0
    # params.not_warmup = True

    GPU = params.gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    torch.cuda.set_device(GPU)
    CUDA = 'cuda:' + str(GPU)

    # output and tensorboard dir
    params.checkpoint_dir = '%s/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    # define logger
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
    logger.add('./logs/train/{}/{}.log'.format(params.name, start_time))
    log_writer_dir = os.path.join('./logs/tb_log', params.name)
    tb_writer = SummaryWriter(log_dir=log_writer_dir)

    params_path = os.path.join(params.checkpoint_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)

    logger.info(params.tag)
         
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224
        
    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default

    # define dataset
    logger.info('\n--- Prepare dataloader ---')
    logger.info('\ttrain with seen domain {}'.format(params.dataset))
    logger.info('\tval with seen domain {}'.format(params.testset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.testset, 'val.json') 
     
    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        
        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':    
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
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

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )

    else:
       raise ValueError('Unknown method')

    model = model.cuda()
 
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume_epoch > 0:
        resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
        tmp = torch.load(resume_file, map_location=CUDA)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])
        logger.info('\tResume the training weight at {} epoch.'.format(start_epoch))
        logger.info('resume checkpoint file dir:{}'.format(resume_file))
    elif params.not_warmup == False: #We also support warmup from pretrained baseline feature, but we never used in our paper
        if params.warmup_file is not None:
            warmup_resume_file = params.warmup_file
        else:
            warmup_resume_file = './checkpoints/Pretrain/399.tar'
        logger.info('load pretrain model checkpoint file dir:{}'.format(warmup_resume_file))
        tmp = torch.load(warmup_resume_file, map_location=CUDA)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    logger.info('Start training...')
    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
    logger.success('Finish training')
