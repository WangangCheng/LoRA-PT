import argparse
import os
import random
import logging
import numpy as np
import time
from torch.fft import fft,ifft
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.UNETR import UNETR
import torch.distributed as dist
from models import criterionsWT
from models.criterions import*
from data.BraTS import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']  = '1'
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'
#os.environ["RANK"] = "0"
#os.environ['WORLD_SIZE'] = '1'

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='Wangangcheng', type=str)
parser.add_argument('--experiment', default='UNETR', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',
                    default='UNETR,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./Datasets/EADC', type=str)
parser.add_argument('--train_dir', default='./Datasets/EADC', type=str)
parser.add_argument('--val_dir', default='./Datasets/EADC', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--val_file', default='valid.txt', type=str)
parser.add_argument('--dataset', default='hippo', type=str)
parser.add_argument('--model_name', default='UNETR', type=str)
# parser.add_argument('--input_C', default=1, type=int)
# parser.add_argument('--input_H', default=197, type=int)
# parser.add_argument('--input_W', default=233, type=int)
# parser.add_argument('--input_D', default=189, type=int)
# parser.add_argument('--crop_H', default=128, type=int)
# parser.add_argument('--crop_W', default=128, type=int)
# parser.add_argument('--crop_D', default=128, type=int)
# parser.add_argument('--output_D', default=155, type=int)
# Training Information
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='softmax_dice2', type=str)
parser.add_argument('--num_cls', default=1, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=1000, type=int)
parser.add_argument('--val_epoch', default=100, type=int)
parser.add_argument('--save_freq', default=500, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)
args = parser.parse_args()

class Finetune_UNETR:
    def __init__(self, resume_path, in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, proj_type, norm_name, res_block, dropout_rate, r):
        self.resume_path = resume_path
        self.r = r
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate
        )
        self.load_model()
        self.W2_parameters = []
        self.W4_parameters = []
        self.W6_parameters = []

    def load_model(self):
        checkpoint = torch.load(self.resume_path, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def compute_new_weights(self):
        weights_linear1 = []
        weights_linear2 = []
        weights_qkv = []
        for name, param in self.model.named_parameters():
            if "linear1" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                reshaped_param = param.data.view(1, 3072, 768)
                weights_linear1.append(reshaped_param)
        if weights_linear1:
            weight_tensor_linear1 = torch.cat(weights_linear1, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
     
        for name, param in self.model.named_parameters():
            if "linear2" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                reshaped_param = param.data.view(1, 768, 3072)
                weights_linear2.append(reshaped_param)
            

        if weights_linear2:
            weight_tensor_linear2 = torch.cat(weights_linear2, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
            
        for name, param in self.model.named_parameters():
            if "qkv" in name and len(param.shape) == 2 and param.numel() == 2304 * 768:
                reshaped_param = param.data.view(3, 768, 768)  
                weights_qkv.append(reshaped_param)
            elif "attn.out_proj.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 768:
                reshaped_param = param.data.view(1, 768, 768)
                weights_qkv.append(reshaped_param)

        if weights_qkv:
            weight_tensor_qkv = torch.cat(weights_qkv, dim=0)
        else:
            print("No weights were found suitable for reshaping and stacking.")
            return None
        # Perform t-SVD decomposition

        D1 = fft(weight_tensor_linear1, dim=0)  # 进行快速傅立叶变换
        D2 = fft(weight_tensor_qkv, dim=0)
        D3 = fft(weight_tensor_linear2, dim=0)
        
        W1 = torch.zeros_like(weight_tensor_linear1, dtype=torch.complex64)
        W2 = torch.zeros_like(weight_tensor_linear1, dtype=torch.complex64)
        
        W3 = torch.zeros_like(weight_tensor_qkv, dtype=torch.complex64)
        W4 = torch.zeros_like(weight_tensor_qkv, dtype=torch.complex64)
        
        W5 = torch.zeros_like(weight_tensor_linear2, dtype=torch.complex64)
        W6 = torch.zeros_like(weight_tensor_linear2, dtype=torch.complex64)

        for i in range(weight_tensor_linear1.shape[0]):
            U_mlp, S_mlp, V_mlp = torch.svd(D1[i])
            S_mlp_diag = torch.diag(S_mlp)

            U1 = U_mlp[:, self.r:].type(torch.complex64)
            S1 = S_mlp_diag[self.r:, self.r:].type(torch.complex64)
            V1 = V_mlp[:, self.r:].type(torch.complex64)

            W1[i] = torch.mm(U1, torch.mm(S1, V1.conj().T))

            U2 = U_mlp[:, :self.r].type(torch.complex64).detach().clone()
            S2 = S_mlp_diag[:self.r, :self.r].type(torch.complex64).detach().clone()
            V2 = V_mlp[:, :self.r].type(torch.complex64).detach().clone()
            

            U2 = nn.Parameter(U2)
            S2 = nn.Parameter(S2)
            V2 = nn.Parameter(V2)
            
            self.W2_parameters.extend([U2, S2, V2])


            W2[i] = torch.mm(U2, torch.mm(S2, V2.conj().T))

        for i in range(weight_tensor_qkv.shape[0]):
            U_qkv, S_qkv, V_qkv = torch.svd(D2[i])
            S_qkv_diag = torch.diag(S_qkv)
            
            U3 = U_qkv[:, self.r:].type(torch.complex64)
            S3 = S_qkv_diag[self.r:, self.r:].type(torch.complex64)
            V3 = V_qkv[:, self.r:].type(torch.complex64)

            W3[i] = torch.mm(U3, torch.mm(S3, V3.conj().T))
            
            U4 = U_qkv[:, :self.r].type(torch.complex64).detach().clone()
            S4 = S_qkv_diag[:self.r, :self.r].type(torch.complex64).detach().clone()
            V4 = V_qkv[:, :self.r].type(torch.complex64).detach().clone()
            
            U4 = nn.Parameter(U4)
            S4 = nn.Parameter(S4)
            V4 = nn.Parameter(V4)
            

            self.W4_parameters.extend([U4, S4, V4])

            W4[i] = torch.mm(U4, torch.mm(S4, V4.conj().T))
            
        for i in range(weight_tensor_linear2.shape[0]):
            U_linear2, S_linear2, V_linear2 = torch.svd(D3[i])
            S_linear2_diag = torch.diag(S_linear2)
            
            U5 = U_linear2[:, self.r:].type(torch.complex64)
            S5 = S_linear2_diag[self.r:, self.r:].type(torch.complex64)
            V5 = V_linear2[:, self.r:].type(torch.complex64)

            W5[i] = torch.mm(U3, torch.mm(S5, V5.conj().T))
            
            U6 = U_linear2[:, :self.r].type(torch.complex64).detach().clone()
            S6 = S_linear2_diag[:self.r, :self.r].type(torch.complex64).detach().clone()
            V6 = V_linear2[:, :self.r].type(torch.complex64).detach().clone()
            
            U6 = nn.Parameter(U6)
            S6 = nn.Parameter(S6)
            V6 = nn.Parameter(V6)
            

            self.W6_parameters.extend([U6, S6, V6])

            W6[i] = torch.mm(U6, torch.mm(S6, V6.conj().T))
            
        self.W1 = ifft(W1,dim=0).real
        self.W2 = ifft(W2,dim=0).real
        self.W3 = ifft(W3,dim=0).real
        self.W4 = ifft(W4,dim=0).real
        self.W5 = ifft(W5,dim=0).real
        self.W6 = ifft(W6,dim=0).real
        return self.W1 + self.W2,self.W3 + self.W4,self.W5 + self.W6
    
    def update_weights(self, W_linear1,W_qkv,W_linear2):
        idx_linear1 = 0
        idx_linear2 = 0
        idx_qkv = 0
        for name, param in self.model.named_parameters():
            if "linear1" in name and len(param.shape) == 2 and param.numel() == 3072 * 768:
                param.data = W_linear1[idx_linear1:idx_linear1+1].reshape(param.shape).float()
                idx_linear1 += 1
            elif "linear2" in name and len(param.shape) == 2 and param.numel() == 768 * 3072:
                param.data = W_linear2[idx_linear2:idx_linear2+1].reshape(param.shape).float()
                idx_linear2 += 1
            elif "qkv" in name and len(param.shape) == 2 and param.numel() == 2304 * 768:
                param.data = W_qkv[idx_qkv:idx_qkv+3].reshape(param.shape).float()
                idx_qkv += 3
            elif "attn.out_proj.weight" in name and len(param.shape) == 2 and param.numel() == 768 * 768:
                param.data = W_qkv[idx_qkv:idx_qkv+1].reshape(param.shape).float()
                idx_qkv += 1
    
    def freeze_parameters(self):
        for name, param in self.model.named_parameters():
            if not any(key in name for key in ["decoder"]):
            #if not any(key in name for key in ["linear1", "linear2", "qkv", "attn.out_proj.weight", "decoder"]):
                param.requires_grad = False
        for param in self.W2_parameters or param in self.W4_parameters or param in self.W6_parameters:
            param.requires_grad = True

    def finetune(self):
        W_linear1,W_qkv,W_linear2 = self.compute_new_weights()
        if W_linear1 is not None:
            self.update_weights(W_linear1,W_qkv,W_linear2)
            self.freeze_parameters()
            
def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
    log_file = log_dir + '.txt' 
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Example usage
    finetuner = Finetune_UNETR(
        resume_path='./checkpoint/UNETR2024-05-23/model_epoch_last.pth',
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
        r=1
    )
    finetuner.finetune()
    model = finetuner.model
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    criterionWT = getattr(criterionsWT, args.criterion)

 
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter()



    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.val_dir, args.val_file)
    val_root = os.path.join(args.root, args.val_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    val_set = BraTS(val_list, val_root, args.mode)

    
    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for val = {}'.format(len(val_set)))


    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,shuffle=True,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)


    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch): 
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        #train

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(x)
            loss,loss_00,loss_01 = criterionWT(output, target)
            reduce_loss = loss.item()
            reduce_loss_00 = loss_00.item()
            reduce_loss_01 = loss_01.item()
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} |0:{:.4f}|1:{:.4f} |'.format(epoch, i, reduce_loss,reduce_loss_00, reduce_loss_01))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
        torch.cuda.empty_cache()
        end_epoch = time.time()

        #val
        if epoch%args.val_epoch==0:
             logging.info('Samples for val = {}'.format(len(val_set)))
             with torch.no_grad():
                 for i, data in enumerate(val_loader):
                     #adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
                     x, target = data
                     x = x.cuda(non_blocking=True)
                     target = target.cuda(non_blocking=True)
                     output = model(x)
                     loss_01 = Dice(output[:, 1, ...], (target == 1).float())
                     #loss_02 = Dice(output[:, 2, ...], (target == 2).float())
                     #loss_03 = Dice(output[:, 3, ...], (target == 3).float())


                     logging.info('Epoch: {}_Iter:{}  Dice: 1:{:.4f}||'
                         .format(epoch, i,  1-loss_01))
        end_epoch = time.time()  
        

        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                or (epoch + 1) % int(args.end_epoch - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss', reduce_loss, epoch)
            #writer.add_scalar('loss_0', reduce_loss_0, epoch)
            #writer.add_scalar('loss_1', reduce_loss_1, epoch)
            writer.add_scalar('loss_00', reduce_loss_00, epoch)
            writer.add_scalar('loss_01', reduce_loss_01, epoch)
            #writer.add_scalar('loss_02', reduce_loss_02, epoch)
            #writer.add_scalar('loss_03', reduce_loss_03, epoch)
   


        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))


    writer.close()

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')




def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
