import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvt_v2 import pvt_v2_b2
# from lib.refinement_module import SelFuseFeature



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





class Net(nn.Module):
    def __init__(self, in_channels,num_classes):
        super().__init__()

        self.backbone = pvt_v2_b2()
        path = "./pvt_v2_b2.pth"
        # path = "/home/ying/Downloads/pvt_v2_b2.pth"
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
        self.transpose1_3 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes, 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))
        self.transpose1_4 = nn.Sequential(nn.ConvTranspose2d(num_classes,num_classes , 2, stride=2),nn.BatchNorm2d(num_classes),nn.ReLU(inplace=True))



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


        pvt = self.backbone(x)
        x1 = pvt[0]    #[64,128,320,512]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
  
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

        glob1 = self.patch1_1(torch.cat((final_patch1_1,x_big_patch[0]),dim=1))
        glob2 = self.patch1_2(torch.cat((final_patch1_2,x_big_patch[1]),dim=1))
        glob3 = self.patch1_3(torch.cat((final_patch1_3,x_big_patch[2]),dim=1))
        glob4 = self.patch1_4(torch.cat((final_patch1_4,x_big_patch[3]),dim=1))

        glob = [glob1,glob2,glob3,glob4]
        glob = self.pool(merge_image_patches(glob))

        final1 = self.predict1(glob)

        return [final,final1]



class SANGRENet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        
        self.encoder = Net(in_channels,num_classes)
    

    def forward(self, x):

        final= self.encoder(x)
       

        return final




