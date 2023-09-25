import torch
import torch.nn as nn
from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

class AE(nn.Module):
    def __init__(self, in_channel=1, out_channel=3):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),  # b, 16, 256, 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 16, 128, 128
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # b, 32, 128, 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 32, 64, 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # b, 64, 64, 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 64, 32, 32
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # b, 16, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channel, kernel_size=4, stride=2, padding=1),  # b, 3, 256, 256
            #nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# model = AE()
# test = torch.randn(1, 3, 256, 256)
# model(test)

class VAE(nn.Module):
    def __init__(self, L_dim = 32, N_dim = 16, F_dim = 16, D_out_dim = 64, in_out = [1, 3]):
        super(VAE, self).__init__()
        self.N_dim = N_dim
        self.RGB_Encoder = RGB_Encoder(in_out[1], F_dim)
        self.Gaussian_Predictor = Gaussian_Predictor(F_dim, N_dim) #try don't use RGB_Encoder
        self.Label_Encoder = Label_Encoder(in_out[0], L_dim)
        self.Decoder_Fusion = Decoder_Fusion(L_dim + N_dim, D_out_dim)
        self.Generator = Generator(D_out_dim, in_out[1])
    
    def forward(self, x, y, train = True):
        label = self.Label_Encoder(x)
        img = self.RGB_Encoder(y)
        if train:
            z, mu, logvar = self.Gaussian_Predictor(img)
        else:
            z = torch.cuda.FloatTensor(x.size(0) ,self.N_dim, label.size(2), label.size(3)).normal_()
            mu = 0
            logvar = 0
        param = self.Decoder_Fusion(label, z)
        out = self.Generator(param)
        return out, mu, logvar