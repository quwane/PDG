import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import DNNclass
from utils import PCC
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# Load the dataset from an HDF5 file

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
model = DNNclass.Holograph_hybrid(args=args, input_dim=10000, hidden_dim=500, dropout_prob=0.3, output_dim=10000)
path = 'best_model_complex.pth'
checkpoint = torch.load(path, map_location='cpu')
print(checkpoint)
# Load the 'phase' tensor directly into the model's phase parameter
model.phase.data = checkpoint['phase']
# Optionally load the rest of the model's state_dict if needed
model_dict = model.state_dict()
state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys() and k != 'phase'}  # Exclude 'phase'
model_dict.update(state_dict)
model.load_state_dict(model_dict)
model = model.cuda()
criterion_pcc = PCC()
criterion_mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam([{'params': model.phase, 'lr': 1e-7}, {'params': model.fc1.parameters(), 'lr': 1e-4}, {'params': model.fc2.parameters(), 'lr': 1e-3}])
Scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=35, gamma=0.8)
num_epochs = 500
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
    dataloader = DataLoader(dataset, batch_size=batch_num, shuffle=False)
    train_loss = 0.0
    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.view(batch_num, 100, 100).cuda()
        labels_batch = labels_batch.view(batch_num, 1, 28, 28)
        img = F.interpolate(labels_batch, size=[100, 100], mode='nearest')
        img = img.view(batch_num, -1)
        img = img.cuda()
        output = model.forward(images_batch)
        output = output.view(batch_num, -1)
        mse_loss = criterion_mse(img, output)
        pcc_loss = 1 - criterion_pcc(img, output)
        loss = 1 * pcc_loss + 3 * mse_loss
        optimizer.zero_grad()
        loss.backward()
        train_loss += pcc_loss.item()
        optimizer.step()
    Scheduler.step()
    loss_epoch = train_loss / len(dataloader)
    print('The %dth epoch, the loss is %4f' %(epoch + 1,loss_epoch))
    if loss_epoch < best_loss:
        save_path = "best_model_hybrid.pth"
        # Save the model's state dictionary to a file
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        best_loss = loss_epoch
        # print("Batch of images shape:", images_batch.shape)
        # print("Batch of labels shape:", labels_batch.shape)
        # break

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images_batch, labels_batch in dataloader:
        images_batch = images_batch.view(batch_num, 100, 100).cuda()
        labels_batch = labels_batch.view(batch_num, 1, 28, 28)
        img = F.interpolate(labels_batch, size=[100, 100], mode='nearest')
        img = img.view(batch_num, -1)
        img = img.cuda()
        outputs = model.forward(images_batch)
        for i in range(batch_num):
            output_image = outputs[i+20].cpu().detach().numpy().reshape(100, 100)  # Convert to NumPy array
            plt.imshow(output_image)
            plt.show()