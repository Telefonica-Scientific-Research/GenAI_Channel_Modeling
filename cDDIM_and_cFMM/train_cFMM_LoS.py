''' 
This script is a code of "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching: 
Fidelity, Efficiency, and Downstream Utility" and it was inspired by the code of
"Generating High Dimensional User-Specific Wireless Channels using Diffusion Models" from
https://arxiv.org/abs/2409.03924.
'''

from tqdm import tqdm
import torch
print(torch.cuda.device_count())
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor
import os
import time

# Set CUDA_VISIBLE_DEVICES=""
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set the SEED for reproducibility
np.random.seed(3)
torch.manual_seed(3)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnetFlowMatching(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=3):
        super(ContextUnetFlowMatching, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 8)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.coord_dim = 3
        self.coord_embed = nn.Linear(self.coord_dim, n_feat)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(1, 8), stride=(1, 8)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = c.float()
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out


def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    Ax = dft_matrix(Nx)
    Ay = dft_matrix(Ny)
    return np.kron(Ay, Ax) # shape: (Nx*Ny, Nx*Ny)

def upa_to_beamspace(H: np.ndarray,
    Nrx_x: int, Nrx_y: int,
    Ntx_x: int, Ntx_y: int) -> np.ndarray:
    """
    H shape: (Nr, Nt)
    where Nr = Nrx_x*Nrx_y and Nt = Ntx_x*Ntx_y
    """
    Ar = upa_dft_codebook(Nrx_x, Nrx_y)
    At = upa_dft_codebook(Ntx_x, Ntx_y)

    Hv = Ar.conj().T @ H @ At
    return Hv


class CustomSionnaDataset(Dataset):
    def __init__(self, data_path, percent_start, percent_end, flag_data_split, save_dir, indices_dir):
        self.data_path = data_path
        self.percent_start = percent_start
        self.percent_end = percent_end
        self.flag_data_split = flag_data_split
        self.idx_start_train = 0
        self.idx_end_train = None
        self.total_length = None
        self.idx_start = None
        self.idx_end = None
        self.indices_dir = indices_dir
        self.save_dir = save_dir # In this directory is where we look for the shuffled indices and where we save the shuffled indices if they do not exist. This is important to ensure that the same shuffling is applied to both training and validation datasets.
        self.load_dataset()
    
    def load_dataset(self):
        np_array = np.load(self.data_path)
        data = np_array["combined_array"][:, :, 0, :, 0, 0, :] 

        print(f"Total number of samples (LoS): {data.shape[0]}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Total number of samples (LoS): {data.shape[0]}\n")

        if os.path.exists(os.path.join(self.indices_dir, 'indices.npy')):
            print(f"Shuffled indices found in {self.indices_dir}. Loading data.")
            indices = np.load(os.path.join(self.indices_dir, 'indices.npy'))

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices loaded from {os.path.join(self.indices_dir, 'indices.npy')}\n")

            np.save(os.path.join(self.save_dir, 'indices.npy'), indices)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices.npy')}")

        else:
            print(f"No shuffled indices found in {self.indices_dir}. Creating shuffled indices and saving them.")
            
            indices = np.arange(np.shape(data)[0])
            np.random.shuffle(indices)

            np.save(os.path.join(self.save_dir, 'indices.npy'), indices)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices.npy')}")

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices created and saved to {os.path.join(self.save_dir, 'indices.npy')}\n")

        data = data[indices]

        self.total_length = np.shape(data)[0]
        self.idx_start = int(self.percent_start * self.total_length)
        if self.percent_end > 1.0:
            self.idx_end = self.percent_end
        else:
            self.idx_end = int(self.percent_end * self.total_length)

        H_set = data[:, :, :, 0]
        coords = data[:, 0, 0, 1:]
        H_set_new = H_set[:,:,:] 

        H_set_new = H_set_new[self.idx_start:self.idx_end,:,:] 
        array1 = np.stack((np.real(H_set_new[:,:,:]), np.imag(H_set_new[:,:,:])), axis=1)
        array2 = array1.copy()

        np.save(os.path.join(self.save_dir, self.flag_data_split+'.npy'), array1)
        print(f"** {self.flag_data_split} data saved to {os.path.join(self.save_dir, self.flag_data_split+'.npy')}")

        for i in range(self.idx_end - self.idx_start) : 
            dft_data = upa_to_beamspace(array2[i,0]+1j*array2[i,1], Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i,0] = np.real(dft_data) # Real part of the DFT
            array1[i,1] = np.imag(dft_data) # Imaginary part of the DFT
        
        self.data = []
        self.labels = []  
        labels_array = coords[self.idx_start:self.idx_end,:] 
        np.save(os.path.join(self.save_dir, self.flag_data_split+'_coords.npy'), labels_array)

        for i in range(array1.shape[0]):
            magnitude = np.sqrt(array1[i, 0, :, :]**2 + array1[i, 1, :, :]**2)

            max_magnitude = np.max(magnitude)

            array1[i, 0, :, :] /= max_magnitude
            array1[i, 1, :, :] /= max_magnitude

            self.data.append(array1[i, :2, :, :]) 
            self.labels.append(labels_array[i]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = ToTensor()(self.data[idx]).float()
        label = self.labels[idx]
        return data_item, label

@torch.no_grad()
def sample_flow_matching_model(
    model,
    cond: torch.Tensor,
    shape: tuple,         # (C, H, W) e.g. (2,4,32)
    device: torch.device,
    guide_w: float = 0.0,
    steps: int = 10
):
    """
    Integrate the learned velocity field v(x,t,cond) with Euler to generate samples.
    Returns: (x_gen, None) to match your previous call pattern.
    - model: your ContextUnetFlowMatching instance (must be in eval mode or will be set)
    - cond: conditioning tensor of shape [n_sample, cond_dim] or [k,cond_dim] (will be repeated if k==1)
    - shape: (C, H, W)
    - device: torch device
    - guide_w: classifier-free guidance weight (0 -> no guidance)
    - steps: number of Euler steps (higher -> better quality, slower)
    """
    model = model.to(device)
    model.eval()

    C, H, W = shape
    cond = cond.to(device, dtype=torch.float32)
    batch_size = cond.shape[0]

    with torch.no_grad():
        x = torch.randn((batch_size, C, H, W), device=device)  # start from Gaussian noise
        dt = 1.0 / float(steps)

        for i in range(steps):
            # current time in [0,1)
            t_val = torch.full((batch_size, 1), float(i) / float(steps), device=device, dtype=torch.float32)

            if guide_w != 0.0:
                null_cond = torch.zeros_like(cond)
                v_uncond = model(x, null_cond, t_val)
                v_cond = model(x, cond, t_val)
                v = v_uncond + guide_w * (v_cond - v_uncond)
            else:
                v = model(x, cond, t_val)

            x = x + v * dt

    return x, None


def flow_matching_loss(model, x0, cond):
    """
    Flow matching: train model to predict dx/dt = (x0 - z)
    """
    B = x0.size(0)
    device = x0.device
    
    # sample time uniformly
    t = torch.rand(B, 1, device=device)  # shape [B,1]
    t_in = t.view(B, 1)                  # input to time embedding
    
    # noise sample
    z = torch.randn_like(x0)

    # Compute the Conditional Flow Matching objective
    # Check eq. (22) and (23) in the paper: https://arxiv.org/abs/2210.02747
    # interpolate x_t
    xt = (1 - t.view(B,1,1,1)) * z + t.view(B,1,1,1) * x0

    # true velocity field
    v_true = x0 - z

    # model prediction
    v_pred = model(xt, cond, t_in)

    return F.mse_loss(v_pred, v_true)

def train():
    # python train_cFMM_LoS.py
    n_epoch = 2000
    batch_size = 100
    steps = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_classes = 3
    n_feat = 256
    lrate = 1e-4
    flag_frequency = 3.5 # It can be either 3.5 or 28 depending on which dataset we want to use
    save_model = False

    logs_main_dir = './logs/'
    if not os.path.exists(logs_main_dir):
        os.makedirs(logs_main_dir)

    # Set timestamp to store in the logs up to minutes
    day_of_experiment = time.strftime("%d_%m_%Y_%H_%M")
    print(f"Day of experiment: {day_of_experiment}")

    train_split_init = 0.0
    train_split_end = 100 #0.7
    val_split_init = 0.7
    val_split_end = 0.9
    save_dir = f'{logs_main_dir}FMM_dataset_{flag_frequency}GHz_LoS_{train_split_init}_{train_split_end}_{day_of_experiment}/'
    ws_test = [0.0] # strength of generative guidance/ it is a guidance weight that controls the influence of the conditioning. See classifier guidance paper for more details

    # Here we point to the directory where the shuffled indices are stored from the training of cDDIM. This is important to ensure that we use the same train/val split as in cDDIM for a fair comparison.
    indices_path = "./logs/CDDIM_dataset_3.5GHz_LoS_0.0_100_28_05_2026_14_13/"

    print(f"** Storing results in {save_dir}")

    # Create save_dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Empty the directory
        os.system(f'rm -rf {save_dir}*')

    # Create a directory in save_dir called "models_x_epoch"
    if not os.path.exists(save_dir+f'models_x_epoch'):
        os.makedirs(save_dir+f'models_x_epoch')

    model = ContextUnetFlowMatching(
        in_channels=2,  
        n_feat=n_feat,
        n_classes=n_classes
    ).to(device)

    # Print the model size
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size} parameters")
    
    # Change the data_path below to use the 3.5GHz or the 28GHz dataset. 
    if flag_frequency == 28:
        data_path = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz"
    else:
        data_path ="../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"

    print("** TRAINING **")
    dataset_train = CustomSionnaDataset(data_path, train_split_init, train_split_end, "train", save_dir, indices_path)
    print("** VALIDATION **")
    dataset_val = CustomSionnaDataset(data_path, val_split_init, val_split_end, "val", save_dir, indices_path)

    # Store all the variables used for training in a .txt file in save_dir for reproducibility
    with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
        f.write(f"n_epoch: {n_epoch}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_classes: {n_classes}\n")
        f.write(f"n_feat: {n_feat}\n")
        f.write(f"lrate: {lrate}\n")
        f.write(f"save_model: {save_model}\n")
        f.write(f"train_split_init: {train_split_init}\n")
        f.write(f"train_split_end: {train_split_end}\n")
        f.write(f"val_split_init: {val_split_init}\n")
        f.write(f"val_split_end: {val_split_end}\n")
        f.write(f"ws_test: {ws_test}\n")
        f.write(f"data_path: {data_path}\n")
        f.write(f"indices_path: {indices_path}\n")
        f.write(f"flag_frequency: {flag_frequency}\n")
        f.write(f"model_size: {model_size}\n")

    # Create a DataLoader for the BerUMaL dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    best_val_loss = float('inf') 
    beg_train_time = time.time()
    for ep in range(n_epoch):
        print(f'** Epoch {ep}/{n_epoch} **')
        model.train()

        pbar = tqdm(dataloader_train)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()

            x = np.transpose(x, (0, 2, 3, 1))
            x = x.to(device, dtype=torch.float32)   
            c = c.to(device, dtype=torch.float32)   

            loss = flow_matching_loss(model, x, c)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # Store the description in a .txt file in save_dir for monitoring
        with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {ep}/{n_epoch}, Training loss: {loss_ema:.4f}\n')

        # Save model every 5000 epochs
        if ep % 100 == 0 :
            model.eval()
            print(f"Evaluating model at epoch {ep}...")
            with torch.no_grad():
                list_mse_losses = []
                for w in ws_test:
                    pbar = tqdm(dataloader_val)
                    start_time = time.time()
                    for x_val, c_val in pbar:
                        x_val = np.transpose(x_val, (0, 2, 3, 1))

                        x_val = x_val.to(device)
                        c_val = c_val.to(device)
                        c_val = c_val.detach().clone().to(device)

                        x_gen, _ = sample_flow_matching_model(
                            model=model,
                            cond=c_val,
                            shape=(2, 4, 32),
                            device=device,
                            guide_w=w,
                            steps=steps
                        )
                        
                        mse = F.mse_loss(x_gen, x_val)
                        pbar.set_description(f"validation loss: {mse.item():.4f}")
                        list_mse_losses.append(mse.item())

                    end_time = time.time()
                    inference_time = end_time - start_time

                    validation_loss = np.mean(list_mse_losses)
                    print(f"  Inference time: {inference_time:.2f} seconds")
                    print(f"  Validation loss (MSE): {validation_loss:.4f}")
                    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                        f.write(f'Epoch {ep}, Validation loss: {validation_loss:.4f}, Inference time: {inference_time:.2f} seconds\n')

                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        print(f"  New best validation loss: {best_val_loss:.4f}")
                        torch.save(model.state_dict(), save_dir + f"model.pth")
                        print('  Saved model at ' + save_dir + f"model.pth")
                        with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                            f.write(f'  New best model saved with validation loss: {best_val_loss:.4f}\n')
                    
                    current_time = time.time()
                    print(f"Total time so far (s): {current_time - beg_train_time:.2f}")
                    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                        f.write(f'--> Total time so far (s): {current_time - beg_train_time:.2f}\n')


if __name__ == "__main__":
    train()

