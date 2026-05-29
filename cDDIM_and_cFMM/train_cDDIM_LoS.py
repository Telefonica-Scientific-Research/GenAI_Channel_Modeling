''' 
This script is a code of "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching: 
Fidelity, Efficiency, and Downstream Utility" and it was inspired by the code of
"Generating High Dimensional User-Specific Wireless Channels using Diffusion Models" from
https://arxiv.org/abs/2409.03924.
'''

from tqdm import tqdm
import torch
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


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=3):
        super(ContextUnet, self).__init__()

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

        # Embedding for the 3D coordinates
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

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed context, time step
        c = c.float()
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Combine embeddings with upsampling
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddim_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDIM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    DDIM_coeff = sqrtmab - torch.sqrt(alpha_t) * torch.sqrt(1 - alphabar_t / alpha_t) # DDIM coef.

    return {
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta,  
        "sqrt_beta_t": sqrt_beta_t,  
        "alphabar_t": alphabar_t,  
        "sqrtab": sqrtab,  
        "sqrtmab": sqrtmab, 
        "DDIM_coeff": DDIM_coeff, 
    }


class DDIM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        ) 
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, c_val, size, device, guide_w = 0.0):
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        context_mask = torch.zeros_like(c_val).to(device)

        # double the batch
        c_val = c_val.repeat(2, 1) # These are the UEs coordinates
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 1. 
        
        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # DDIM step (deterministic, no random noise added)
            eps = self.nn_model(x_i, c_val, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.DDIM_coeff[i])

            # if i % 20 == 0 or i == self.n_T or i < 8:
            #     x_i_store.append(x_i.detach().cpu().numpy())

        # x_i_store = np.array(x_i_store)
        return x_i, x_i

def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    Ax = dft_matrix(Nx)
    Ay = dft_matrix(Ny)
    # planar DFT codebook
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
        # In self.data_path there is a zip file with a npy file inside
        # Load the npy file
        np_array = np.load(self.data_path)
        # Extract the array (since it's stored with a key). Take the first element from the 256 subcarriers
        data = np_array["combined_array"][:, :, 0, :, 0, 0, :] # (num_samples, num_rx, num_tx_ant, coordinates)

        # Print total number of samples and write to training_log.txt
        print(f"Total number of samples (LoS): {data.shape[0]}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Total number of samples (LoS): {data.shape[0]}\n")

        # Check if indices.npy exists in save_dir. If it does, load the indices and shuffle data accordingly. If it does not, create the shuffled indices and save them to save_dir.
        if os.path.exists(os.path.join(self.indices_dir, 'indices.npy')):
            print(f"Shuffled indices found in {self.indices_dir}. Loading data.")
            indices = np.load(os.path.join(self.indices_dir, 'indices.npy'))

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices loaded from {os.path.join(self.indices_dir, 'indices.npy')}\n")

            # Store the shuffled indices to save_dir
            np.save(os.path.join(self.save_dir, 'indices.npy'), indices)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices.npy')}")

        else:
            print(f"No shuffled indices found in {self.indices_dir}. Creating shuffled indices and saving them.")
            
            indices = np.arange(np.shape(data)[0])
            # Shuffle indices
            np.random.shuffle(indices)

            # Store the shuffled indices to save_dir
            np.save(os.path.join(self.save_dir, 'indices.npy'), indices)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices.npy')}")

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices created and saved to {os.path.join(self.save_dir, 'indices.npy')}\n")

        # Shuffle data using the shuffled indices
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

        # Store array1 as npy in save_dir for Firdous
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
            # Calculating the magnitude
            magnitude = np.sqrt(array1[i, 0, :, :]**2 + array1[i, 1, :, :]**2)

            # Finding the maximum magnitude
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


def train():
    # (cDDIM) python train_cDDIM_LoS.py 
    n_epoch = 3000
    batch_size = 100
    n_T = 150
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
    train_split_end =  100 #0.7
    val_split_init = 0.7
    val_split_end = 0.9
    save_dir = f'{logs_main_dir}CDDIM_dataset_{flag_frequency}GHz_LoS_{train_split_init}_{train_split_end}_{day_of_experiment}/'
    
    # Here we point to the directory where the shuffled indices are stored from the training of cDDIM. This is important to ensure that we use the same train/val split as in cDDIM for a fair comparison.
    indices_path = save_dir #"/datasets/CDDIM_dataset_3.5GHz_LoS_0.0_200_nT10/"
    
    ws_test = [0.0] # strength of generative guidance/ it is a guidance weight that controls the influence of the conditioning. See classifier guidance paper for more details

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

    ddim = DDIM(
        nn_model=ContextUnet(in_channels=2, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    )
    ddim.to(device)

    # Print the model size
    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model size: {model_size} parameters")

    # Change the data_path below to use the 3.5GHz or the 28GHz dataset. 
    if flag_frequency == 28:
        data_path = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz"
    else:
        data_path = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"

    print("** TRAINING **")
    dataset_train = CustomSionnaDataset(data_path, train_split_init, train_split_end, "train", save_dir, indices_path)
    print("** VALIDATION **")
    dataset_val = CustomSionnaDataset(data_path, val_split_init, val_split_end, "val", save_dir, indices_path)

    # Store all the variables used for training in a .txt file in save_dir for reproducibility
    with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
        f.write(f"n_epoch: {n_epoch}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"n_T: {n_T}\n")
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

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)

    optim = torch.optim.Adam(ddim.parameters(), lr=lrate)

    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    for ep in range(n_epoch):
        print(f'** Epoch {ep}/{n_epoch} **')
        ddim.train()

        pbar = tqdm(dataloader_train)
        loss_ema = None
        for x, c in pbar:
            # c are the positions of the UEs
            # x is the channel data
            optim.zero_grad()
            x = np.transpose(x, (0, 2, 3, 1))
            x = x.to(device)
            c = c.to(device)
            c = torch.tensor(c, dtype=torch.float32).to(device)

            loss = ddim(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Training loss: {loss_ema:.4f}")
            optim.step()
        
        # Store the description in a .txt file in save_dir for monitoring
        with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {ep}/{n_epoch}, Training loss: {loss_ema:.4f}\n')

        # # Store the model at every epoch with the epoch identifier
        # torch.save(ddim.state_dict(), save_dir + f"models_x_epoch/model_epoch_{ep}.pth")
        # print(f"Saved model at epoch {ep} to " + save_dir + f"models_x_epoch/model_epoch_{ep}.pth")

        # Save model every 200 epochs
        if ep % 200 == 0 :
            ddim.eval()
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

                        # Generate samples using DDIM
                        x_gen, _ = ddim.sample(batch_size, c_val, (2, 4, 32), device, guide_w=w)

                        mse = F.mse_loss(x_gen, x_val)
                        pbar.set_description(f"validation loss: {mse.item():.4f}")
                        list_mse_losses.append(mse.item())
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                    validation_loss = np.mean(list_mse_losses)

                    # Print inference time and validation loss
                    print(f"  Inference time: {inference_time:.2f} seconds")
                    print(f"  Validation loss (MSE): {validation_loss:.4f}")
                    # Store the description in a .txt file in save_dir for monitoring
                    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                        f.write(f'Epoch {ep}, Validation loss: {validation_loss:.4f}, Inference time: {inference_time:.2f} seconds\n')

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    print(f"  New best validation loss: {best_val_loss:.4f}")
                    torch.save(ddim.state_dict(), save_dir + f"model.pth")
                    print('  Saved model at ' + save_dir + f"model.pth")
                    # Store the description in a .txt file in save_dir for monitoring
                    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                        f.write(f'  New best model saved with validation loss: {best_val_loss:.4f}\n')

if __name__ == "__main__":
    train()

