import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import ConvBnRelu2d, StackEncoder, StackDecoder

# 1024*1024
class UNet1024(nn.Module):

    def __init__(self, in_shape) -> None:
        super(UNet1024, self).__init__()
        C, H, W = in_shape
        # assert(C == 3)

        #1024
        self.down1 = StackEncoder(C, 24, kernel_size=3)    # 512
        self.down2 = StackEncoder(24, 64, kernel_size=3)   # 256
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 128
        self.down4 = StackEncoder(128, 256, kernel_size=3) # 64
        self.down5 = StackEncoder(256, 512, kernel_size=3) # 32
        self.down6 = StackEncoder(512, 768, kernel_size=3) # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3)   # 16
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)   # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)   # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)    # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)      # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)      # 512
        self.classify = nn.Conv2d(24, 2, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x                         # print('x    ', x.size()) 

        down1, out = self.down1(out)    # print('down1', down1.size()) # 256
        down2, out = self.down2(out)    # print('down2', down2.size()) # 128
        down3, out = self.down3(out)    # print('down3', down3.size()) # 64
        down4, out = self.down4(out)    # print('down4', down4.size()) # 32
        down5, out = self.down5(out)    # print('down5', down5.size()) # 16
        down6, out = self.down6(out)    # print('down6', down6.size()) # 8
                                        # print('out  ', out.size())

        out = self.center(out)          
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        # 1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)

        return out


# 512*512
class UNet512 (nn.Module):
    def __init__(self, in_shape):
        super(UNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out


# 256x256
class UNet256 (nn.Module):
    def __init__(self, in_shape):
        super(UNet256, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #256
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

if __name__ == '__main__':
    net = UNet1024([3, 1024, 1024])
    input = torch.rand(1, 3, 1024, 1024)
    print(input.size())
    print(type(input))
    output = net(input)

    pass