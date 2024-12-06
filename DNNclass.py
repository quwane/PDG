import torch
import numpy as np
import math
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm
import shutil
import torch.nn as nn
from scipy.ndimage import gaussian_filter

def diffraction(wave, trans_func, scattering):
    padding_size = wave.shape[1] // 2
    wave = F.pad(wave, pad=[padding_size, padding_size, padding_size, padding_size])
    wave = torch.squeeze(wave)
    wave_f = torch.fft.fft2(torch.fft.fftshift(wave))
    wave_f *= (trans_func)
    wave = torch.fft.ifftshift(torch.fft.ifft2(wave_f))
    wave = wave[:, wave.shape[1] // 4: (wave.shape[1] // 4 + 2 * padding_size),
            wave.shape[1] // 4: (wave.shape[1] // 4 + 2 * padding_size)]
    return wave

def modulation(args, wave, phase1):
    phase = args.scale * math.pi * (torch.sin(args.alpha * phase1) + 1)
    phase = phase % (2 * math.pi)
    if args.phase_noise is True:
        phase = phase + np.sqrt(args.phase_noise_level) * torch.randn(phase.size())
    test = 2
    if test == 1:
        wave = wave * torch.exp(1.0j * 2 * np.pi *torch.sigmoid(phase1))
    elif test == 2:
        wave = wave * torch.exp(1.0j * phase)
    elif test ==3 :
        wave = wave * torch.exp(1.0j * phase1)
    return wave

def transfer_kernel(z, wavelength, N, pixel_size, bandlimit=True, gpu=True):
        k = 2.0 * np.pi / wavelength
        length_screen = pixel_size * N
        fs = 1.0 / (2 * length_screen)  # '2' comes from zero padding
        fx = np.linspace(-1 / (2 * pixel_size), (1 / (2 * pixel_size) - fs), 2 * N)
        fy = fx
        Fx, Fy = np.meshgrid(fx, fy)
        ph0 = Fx ** 2 + Fy ** 2
        ph = np.exp(1.0j * z * np.sqrt(k ** 2 - np.multiply(4 * np.pi ** 2, ph0)))
        if bandlimit:
            fxlimit = 1 / np.sqrt(1 + (2 * fs * z) ** 2) / wavelength
            fylimit = fxlimit
            ph[np.abs(Fx) > fxlimit] = 0
            ph[np.abs(Fy) > fylimit] = 0

        h = np.fft.fftshift(ph)
        h = torch.from_numpy(h)
        if gpu:
            h = h.cuda()
        return h

class Holograph(nn.Module):
    def __init__(self, args):
        super(Holograph, self).__init__()
        self.args = args
        self.scattering = None
        self.h = transfer_kernel(z=self.args.distance, wavelength=self.args.wavelength,N=self.args.n_numx, pixel_size=self.args.pixel_size,
                                 bandlimit=True, gpu=True)
        self.phase = nn.Parameter(0.0 * torch.randn(args.num_layers, args.n_numx, args.n_numy), requires_grad=True)
    def forward(self, input):
        for idx in range(self.args.num_layers):
            input = modulation(self.args, input, self.phase[idx, :, :])
            input = diffraction(input, self.h, self.scattering)
            if self.args.detector_noise is True:
                input = torch.square(torch.abs(input)).view(input.size(0), -1)
                input_std = torch.std(input, dim=1, keepdim=True)
                noise = np.sqrt(self.args.det_noise_level) * torch.randn(input.size())
                input = torch.abs(input / input_std + noise)
                input = torch.sqrt(input).view(input.size(0, self.args.n_numx, self.args.n_numy))
            else:
                # input = torch.abs(input).view(input.size(0), -1)
                # input_std = torch.std(input, dim=1, keepdim=True)
                # input = torch.abs(input / input_std)
                # input = torch.sqrt(input).view(input.size(0), self.args.n_numx, self.args.n_numy)
                input = torch.abs(input)
        output = torch.square(torch.abs(input))
        return output

class ParaHolograph(nn.Module):
    def __init__(self, args):
        super(Holograph, self).__init__()
        self.args = args
        self.scattering = None
        self.h = transfer_kernel(z=self.args.distance, wavelength=self.args.wavelength,N=self.args.n_numx, pixel_size=self.args.pixel_size,
                                 bandlimit=True, gpu=True)
        self.phase = nn.Parameter(0.0 * torch.randn(args.num_layers, args.n_numx, args.n_numy), requires_grad=True)
        self.weight = nn.Parameter(torch.randn(args.num_layer - 1), require_grad = True)
    def forward(self, input_or):
        for idx in range(self.args.num_layers):
            input = modulation(self.args, input_or, self.phase[idx, :, :])
            input = diffraction(input, self.h, self.scattering)
            if self.args.detector_noise is True:
                input = torch.square(torch.abs(input)).view(input.size(0), -1)
                input_std = torch.std(input, dim=1, keepdim=True)
                noise = np.sqrt(self.args.det_noise_level) * torch.randn(input.size())
                input = torch.abs(input / input_std + noise)
                input = torch.sqrt(input).view(input.size(0, self.args.n_numx, self.args.n_numy))
            else:
                # input = torch.abs(input).view(input.size(0), -1)
                # input_std = torch.std(input, dim=1, keepdim=True)
                # input = torch.abs(input / input_std)
                # input = torch.sqrt(input).view(input.size(0), self.args.n_numx, self.args.n_numy)
                input = torch.abs(input)
            max_feature, _ = input.view(128, -1).max(dim=1)
            input = input / max_feature
            input = input.view(input_or.shape)
            if idx == 0:
                output_temp = input
            else:
                output_temp = self.weight[idx] * input + output_temp
        output_temp = torch.sigmoid(output_temp)
        output_temp = modulation(self.args, output_temp, self.phase[idx, :, :])
        output_temp = diffraction(output_temp, self.h, self.scattering)
        output = torch.square(torch.abs(output_temp))
        return output
    
class Holograph_hybrid(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, dropout_prob, output_dim):
        super(Holograph_hybrid, self).__init__()
        self.args = args
        self.scattering = None
        self.h = transfer_kernel(z=self.args.distance, wavelength=self.args.wavelength,N=self.args.n_numx, pixel_size=self.args.pixel_size,
                                 bandlimit=True, gpu=True)
        self.phase = nn.Parameter(0.0 * torch.randn(args.num_layers, args.n_numx, args.n_numy), requires_grad=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with specified probability
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Hidden to output layer
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        for idx in range(self.args.num_layers):
            self.args.alpha = idx * 1.5
            input = modulation(self.args, input, self.phase[idx, :, :])
            input = diffraction(input, self.h, self.scattering)
            if self.args.detector_noise is True:
                input = torch.square(torch.abs(input)).view(input.size(0), -1)
                input_std = torch.std(input, dim=1, keepdim=True)
                noise = np.sqrt(self.args.det_noise_level) * torch.randn(input.size())
                input = torch.abs(input / input_std + noise)
                input = torch.sqrt(input).view(input.size(0, self.args.n_numx, self.args.n_numy))
            else:
                # input = torch.abs(input).view(input.size(0), -1)
                # input_std = torch.std(input, dim=1, keepdim=True)
                # input = torch.abs(input / input_std)
                # input = torch.sqrt(input).view(input.size(0), self.args.n_numx, self.args.n_numy)
                input = torch.abs(input)
        output = torch.square(torch.abs(input))
        output = output.view(output.shape[0], -1)
        output_max, _  = torch.max(output, dim=1, keepdim=True)
        output = torch.abs(output / output_max)
        # output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
        out = self.fc1(output)  # First layer
        out = self.relu(out)  # ReLU activation
        # out = self.dropout(out)  # Apply dropout during training
        # out = self.fc3(out)  # First layer
        # out = self.relu(out)
        # out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))  # Output layer
        return out
    