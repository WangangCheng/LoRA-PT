import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch, random
import matplotlib.pyplot as plt

MIN_BOUND = -1000.0
MAX_BOUND = 400.0




def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
   # plt.show(net)
    print('Total number of parameters: %d' % num_params)
