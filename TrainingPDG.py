import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import DNNclass
import os
from utils import PCC
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
# Load the dataset from an HDF5 file
output_dir = "model_outputs_fashion"
target_scale = 100
class Parameters:
    def __init__(self):
        # diffraction neural network parameters
        self.distance = 15e-2
        self.wavelength = 532e-9
        self.n_numx = target_scale
        self.n_numy = target_scale
        self.num_layers = 4
        self.scale = 5
        self.alpha = 2
        self.pixel_size = 24e-6
        self.phase_noise = False
        self.phase_noise_level = 1e-3
        self.detector_noise = False
        self.det_noise_level = 1e-2
        # MNIST specific parameters
        self.beta = 5
args = Parameters()
model = DNNclass.Holograph(args=args)
model = model.cuda()
image_counter = 0
criterion_pcc = PCC()
criterion_mse = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
Scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=35, gamma=0.8)
num_epochs = 1000
best_loss = float('inf')
# Example: iterate through the DataLoader
for epoch in range(num_epochs):
    with h5py.File('dataset_len3.h5', 'r') as hdf:
    # Load the dataset from the file
        images = hdf['inputs'][:]  # Assuming 'images' dataset exists
        labels = hdf['ground_truths'][:]  # Assuming 'labels' dataset exists
# Convert the datasets to PyTorch tensors
    images_real =  images.real
    images_imag =  images.imag
    images_tensors_real = torch.Tensor(images_real) 
    images_tensors_imag = torch.Tensor(images_imag) # Convert to tensor
    images_tensor = torch.complex(images_tensors_real, images_tensors_imag)
    labels_tensor = torch.Tensor(labels) # Convert labels to tensor and cast to long (for classification)
    # Create a TensorDataset for PyTorch
    dataset = TensorDataset(images_tensor, labels_tensor)
    # Create a DataLoader for batching and shuffling
    batch_num = 256
    dataloader = DataLoader(dataset, batch_size=batch_num, shuffle=True)
    train_loss = 0.0
    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.view(batch_num, 100, 100).cuda()
        # images_batch = torch.abs(images_batch)
        labels_batch = labels_batch.view(batch_num, 1, 28, 28)
        img = F.interpolate(labels_batch, size=[100, 100], mode='nearest')
        img = img.view(batch_num, -1)
        img = img.cuda()
        output = model.forward(images_batch)
        output = output.view(batch_num, -1)
        mse_loss = criterion_mse(img, output)
        pcc_loss = 1 - criterion_pcc(img, output)
        loss = pcc_loss + 0.5 * mse_loss
        optimizer.zero_grad()
        loss.backward()
        train_loss += pcc_loss.item()
        optimizer.step()
        # if epoch == 10 :
        #     output = output.view(batch_num, 100, 100)
        #     for i in range(batch_num):
        #         output_image = output[i].cpu().detach().numpy()  # Convert to NumPy array

        #         # Normalize the output image to [0, 255] for saving as image
        #         output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
        #         output_image = output_image.astype('uint8')  # Convert to uint8 for saving

        #         # Convert to PIL Image and save
        #         img = Image.fromarray(output_image)
        #         img.save(os.path.join(output_dir, f"output_{image_counter}.bmp"))  # Save as BMP or other formats
        #         image_counter += 1
    Scheduler.step()
    loss_epoch = train_loss / len(dataloader)
    print('The %dth epoch, the loss is %4f' %(epoch + 1,loss_epoch))
    if loss_epoch < best_loss:
        save_path = "best_model_complex.pth"
        # Save the model's state dictionary to a file
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        best_loss = loss_epoch
        # print("Batch of images shape:", images_batch.shape)
        # print("Batch of labels shape:", labels_batch.shape)
        # break