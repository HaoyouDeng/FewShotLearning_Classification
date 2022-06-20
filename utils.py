import torch
import numpy as np
import torch.nn.init as init
import math

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun