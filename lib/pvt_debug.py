import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvt_v2 import pvt_v2_b2
from lib.refinement_module import SelFuseFeature


class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src
       
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hxin + hx1d 
        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi, psi

class fusion_block(nn.Module):
    def __init__(self,in_channels,mid):
        super(fusion_block,self).__init__()
        
        
        self.contex = RSU7(in_channels,mid,in_channels)
        self.conv1  = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1  = nn.BatchNorm2d(in_channels)
        self.relu   = nn.ReLU(True)

        self.conv2  = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2  = nn.BatchNorm2d(in_channels)
        self.relu   = nn.ReLU(True)
        
    def forward(self, x):
        x_context = self.contex(x)

        x_conv = (self.norm1(self.conv1(x)))

        x_cat = torch.cat((x_context,x_conv),dim=1)
        # x_cat = x_context*x_conv
        out= self.relu(self.norm2(self.conv2(x_cat)))
        out = x_conv + out
        return out


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, out_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        length  = len(bins)
        channel = length * reduction_dim
        concat_dim = in_dim + channel
        self.final = nn.Sequential(
            nn.Conv2d(concat_dim, out_dim, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        op=0
        for f in self.features:
            op+=1
            # print(str(op),f(x).shape)
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        concat_dim = torch.cat(out, 1)
        final_out =self.final(concat_dim)
        return final_out



def split_tensor_into_4_patches(f_global):
    """
    Split a tensor into 4 equal patches.

    Parameters:
    - f_global: Tensor of shape (n, c, 512, 512), the global feature maps for the batch.

    Returns:
    - Tensor of shape [(n, c, 256, 256),
    (n, c, 256, 256),
    (n, c, 256, 256),
    (n, c, 256, 256)], where each item in the batch is split into 4 patches.
    """
    # First, ensure that the spatial dimensions are divisible by 2 to get equal patches
    assert f_global.size(2) % 2 == 0 and f_global.size(3) % 2 == 0, "Spatial dimensions must be divisible by 2."
    
    # Number of patches along height and width
    num_patches_height = f_global.size(2) // 2
    num_patches_width = f_global.size(3) // 2
    
    # Reshape to get the 4 patches. We'll split the tensor into 4 along the height and width
    # The resulting shape will be (n, c, 2, 256, 2, 256)
    patches = f_global.reshape(f_global.shape[0], f_global.shape[1], 2, num_patches_height, 2, num_patches_width)
    
    # Permute to bring the patch indices together and merge them
    # The new shape will be (n, 2*2, c, 256, 256) -> (n, 4, c, 256, 256)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(f_global.shape[0], 4, f_global.shape[1], num_patches_height, num_patches_width)
    k = []
    for i in range(4):
      k.append(patches[:,i,:,:])
    return k

def merge_image_patches(patches):
    """
    Merge four image patches into a whole image.

    Parameters:
    - patches: A list containing four patches of dimension [1, 4, 256, 256]. The patches are in the order:
      top-left, top-right, bottom-left, bottom-right of the original image.

    Returns:
    - The whole image of dimension (1, 4, 512, 512) obtained by merging the patches.
    """
    # Check that we have exactly four patches
    assert len(patches) == 4, "There must be exactly four patches."
    
    # Concatenate the top patches (top-left and top-right) along the width
    top_half = torch.cat((patches[0], patches[1]), dim=3)
    
    # Concatenate the bottom patches (bottom-left and bottom-right) along the width
    bottom_half = torch.cat((patches[2], patches[3]), dim=3)
    
    # Concatenate the top and bottom halves along the height to get the whole image
    whole_image = torch.cat((top_half, bottom_half), dim=2)

    return whole_image





class Encoder(nn.Module):
    def __init__(self, in_channels,num_classes):
        super().__init__()

        self.backbone = pvt_v2_b2()
        # path = "./pvt_v2_b1.pth"
        path = "/home/ying/Downloads/pvt_v2_b2.pth"
        # path ="/data/home/eey469/pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        

        self.outconvb = nn.Conv2d(in_channels,3,3,padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norminitial = nn.BatchNorm2d(3)
        self.reluinitial = nn.ReLU(inplace=True)

        self.patch1_1 = nn.Sequential(nn.Conv2d(num_classes+3,num_classes+3,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(num_classes+3),nn.ReLU(inplace=True))
        self.patch1_2 = nn.Sequential(nn.Conv2d(num_classes+3,num_classes+3,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(num_classes+3),nn.ReLU(inplace=True))
        self.patch1_3 = nn.Sequential(nn.Conv2d(num_classes+3,num_classes+3,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(num_classes+3),nn.ReLU(inplace=True))
        self.patch1_4 = nn.Sequential(nn.Conv2d(num_classes+3,num_classes+3,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(num_classes+3),nn.ReLU(inplace=True))

        # self.patch2_1 = nn.Conv2d(12,3,kernel_size=3,stride=1,padding=1)
        # self.patch2_2 = nn.Conv2d(12,3,kernel_size=3,stride=1,padding=1)
        # self.patch2_3 = nn.Conv2d(12,3,kernel_size=3,stride=1,padding=1)
        # self.patch2_4 = nn.Conv2d(12,3,kernel_size=3,stride=1,padding=1)

        self.transpose1_1 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes , 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))
        self.transpose1_2 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes , 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))
        self.transpose1_3 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes , 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))
        self.transpose1_4 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes, 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))



        self.r_linear5 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, stride=8, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.r_linear4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.r_linear3 = nn.Sequential(nn.Conv2d(128, 320, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(320), nn.ReLU(inplace=True))
        self.r_linear2 = nn.Sequential(nn.Conv2d(320, 512, kernel_size=1, stride=2, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        
        self.linear5 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(640, 320, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(320), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.linear2_5 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear2_4 = nn.Sequential(nn.Conv2d(320, 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear2_3 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear2_2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.predict = nn.Conv2d(32*3, num_classes, kernel_size=1, stride=1, padding=0)
      
        self.predict1 = nn.Conv2d(num_classes+3, num_classes, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.drop = nn.Dropout(.2)
    def forward(self, x):
        x     =self.outconvb(x)

        x_big =self.upscore2(x)
        x_big = self.norminitial(x_big)
        x_big = self.reluinitial(x_big)

        x_1_sup = self.r_linear5(x_big)
        x_2_sup = self.r_linear4(x_1_sup)
        x_3_sup = self.r_linear3(x_2_sup)
        x_4_sup = self.r_linear2(x_3_sup)

        # print(x_1_sup.shape,x_2_sup.shape,x_3_sup.shape,x_4_sup.shape) 
        # torch.Size([8, 64, 128, 128]) 
        # torch.Size([8, 128, 64, 64]) 
        # torch.Size([8, 320, 32, 32]) 
        # torch.Size([8, 512, 16, 16])

        # x=self.norminitial(x)
        # x=self.reluinitial(x)

        pvt = self.backbone(x)
        x1 = pvt[0]    #[64,128,320,512]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print('x2',x2.shape)
        x1 = torch.cat((x1,x_1_sup),dim=1)
        x2 = torch.cat((x2,x_2_sup),dim=1)
        x3 = torch.cat((x3,x_3_sup),dim=1)
        x4 = torch.cat((x4,x_4_sup),dim=1)

        x1 = self.linear2(x1) + x_1_sup
        x1 = self.drop(x1)
        x2 = self.linear3(x2) + x_2_sup
        x2 = self.drop(x2)
        x3 = self.linear4(x3) + x_3_sup
        x3 = self.drop(x3)
        x4 = self.linear5(x4) + x_4_sup
        x4 = self.drop(x4)

        x1 = self.linear2_2(x1)
        x2 = self.linear2_3(x2)
        x3 = self.linear2_4(x3)
        x4 = self.linear2_5(x4)

        x1 = F.interpolate(x1, size=128*2, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=128*2, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=128*2, mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=128*2, mode='bilinear', align_corners=True)

        final = self.predict(torch.cat((x1*x2,x1*x2*x3,x1*x2*x3*x4),dim=1)) # 9,256,256


        x_big_patch = split_tensor_into_4_patches(x_big)
        
        final_patch = split_tensor_into_4_patches(final)
        final_patch1_1 = self.transpose1_1(final_patch[0])
        final_patch1_2 = self.transpose1_2(final_patch[1])
        final_patch1_3 = self.transpose1_3(final_patch[2])
        final_patch1_4 = self.transpose1_4(final_patch[3])

        # print(final_patch[0].shape)
        # print(final_patch1_4.shape)
        glob1 = self.patch1_1(torch.cat((final_patch1_1,x_big_patch[0]),dim=1))
        glob2 = self.patch1_2(torch.cat((final_patch1_2,x_big_patch[1]),dim=1))
        glob3 = self.patch1_3(torch.cat((final_patch1_3,x_big_patch[2]),dim=1))
        glob4 = self.patch1_4(torch.cat((final_patch1_4,x_big_patch[3]),dim=1))

        glob = [glob1,glob2,glob3,glob4]
        glob = self.pool(merge_image_patches(glob))

        final1 = self.predict1(glob)

        return [final,final1]



class PvtUNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        
        self.encoder = Encoder(in_channels,num_classes)
    

    def forward(self, x):

        final= self.encoder(x)
       

        return final




