from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time


import numpy as np
from tqdm import tqdm
from medpy.metric import dc,hd95
from scipy.ndimage import zoom

from utils.utils import powerset
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from lib.net import SANGRENet


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=18, help="batch size") #12
parser.add_argument("--lr", default=0.0003, help="learning rate")
parser.add_argument("--max_epochs", default=300)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')   
parser.add_argument('--n_skip',type=int,default=1000)         
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.is_pretrain = True
args.exp = 'pred_32h_correct_loss__' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'Transvit_ref_before_sup')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_'+str(args.img_size)
snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

current_time = time.strftime("%H%M%S")
print("The current time is", current_time)
snapshot_path = 'checkpoint' +'_run'+current_time
    
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)
     



#############################################################      

net = SANGRENet(in_channels=1,num_classes=4).cuda()

from torchinfo import summary
#    summary(model, (1, 3, 352, 352))
from thop import profile
import torch
input = torch.randn(1, 1, 256, 256).to('cuda')
macs, params = profile(net, inputs=(input,))
print('macs:', macs / 1000000000)
print('params:', params / 1000000)

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)






if args.n_gpu > 1:
    net = nn.DataParallel(net)

net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)

# save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.8
Best_dcs_th = 0.865

logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.00001)
#optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

def val():
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]

        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(0).cpu().detach().numpy()
        
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3) # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        P = net(val_image_batch)
        #print(len(P))

        val_outputs = 0.0
        for idx in range(len(P)):
            val_outputs += P[idx]
        
        val_outputs = torch.softmax(val_outputs, dim=1)

        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    #print("val avg_dsc: %f" % (performance))
    return performance


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    
bd_loss=BoundaryDoULoss(args.num_classes)

# l = [0, 1, 2, 3]
# ss = [x for x in powerset(l)] # for mutation
ss = [[0],[1]] # for only four-stage loss, no mutation
# print(ss)
    
dice=[]
for epoch in iterator:
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        
        P = net(image_batch)
        loss = 0.0
        lc1, lc2 = 0.3, 0.7
                  
   
        iout = 0.0
     
        for idx in P:
            iout += idx
        loss_ce = ce_loss(iout, label_batch[:].long())
        loss_dice = dice_loss(iout, label_batch, softmax=True)
        loss += (lc1 * loss_ce + lc2 * loss_dice) 
       
        # loss += loss_bd
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 
        lr_=base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

        iter_num = iter_num + 1
        if iter_num%50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    
    avg_dcs = val()
        
    if avg_dcs >= Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        dice.append(avg_test_dcs)
        print(dice)
        Best_dcs = avg_dcs

    save_interval = 25  # int(max_epoch/6)
    if epoch > int(args.max_epochs / 3) and (epoch + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(net.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        print(avg_test_dcs)
        
    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        iterator.close()
        break

avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
print("test avg_dsc: %f" % (avg_test_dcs))
print(avg_test_dcs)
logging.info("test avg_dsc: %f" % (avg_test_dcs))
Test_Accuracy.append(avg_test_dcs)  

