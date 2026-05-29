''' 
This script is a code of "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching: 
Fidelity, Efficiency, and Downstream Utility" and it was inspired by the code of
"Generating High Dimensional User-Specific Wireless Channels using Diffusion Models" from
https://arxiv.org/abs/2409.03924.
'''

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.cm as cm
import time
from scipy.io import savemat, loadmat
import sys
import os
import time
import seaborn as sns
import pandas as pd
import trimesh
from skimage.draw import polygon
import mitsuba as mi
from sklearn.metrics.pairwise import cosine_similarity
from upa_beam_distance import compute_upa_peak_beam_distance

mi.set_variant("scalar_rgb")  # "cuda_ad_rgb" for GPU or "scalar_rgb" if you don't have CUDA

# Set the SEED for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Set CUDA_VISIBLE_DEVICES=""
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
train_split_init = 0.0
train_split_end = 200 #0.7
val_split_init = 0.7
val_split_end = 0.9
test_split_init = 0.9
test_split_end = 1.0

num_samples_2_plot = 100000 # Once we generated this many samples, we stop generating more

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


def rasterize_mesh_to_heightmap(mesh, H, W, xmin, xmax, ymin, ymax):
    """
    Rasterize a trimesh mesh into a 2D height map.
    
    mesh: trimesh.Trimesh
    H, W: height/width of output image
    xmin, xmax, ymin, ymax: scene bounds
    """
    height_map = np.full((H, W), -np.inf)

    # Iterate over faces
    for face in mesh.faces:
        verts = mesh.vertices[face]  # shape (3, 3): x, y, z

        # Project XY onto 2D pixel grid
        px = ((verts[:, 0] - xmin) / (xmax - xmin) * (W - 1)).astype(int)
        py = ((verts[:, 1] - ymin) / (ymax - ymin) * (H - 1)).astype(int)

        # Fill triangle polygon in pixel space
        rr, cc = polygon(py, px, (H, W))

        # Assign max height (z) of triangle to polygon area
        z_val = verts[:, 2].max()
        height_map[rr, cc] = np.maximum(height_map[rr, cc], z_val)

    # Replace unfilled values with 0
    height_map = np.where(np.isfinite(height_map), height_map, 0.0)
    return height_map

def generate_H_test(ddim, flag_frequency, flag_data_split, device, save_dir, batch_size=5000, n_sample=5000, ws_test=[0.0]):
    list_distances = []
    list_distances_upa = []
    indices_path = save_dir

    hist_bins = 32
    hist_pred = torch.zeros(hist_bins, device=device)
    hist_gt = torch.zeros(hist_bins, device=device)

    print("** TESTING **")
    # Change the data_path below to use the 3.5GHz or the 28GHz dataset. 
    if flag_frequency == 28:
        data_path = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz"
    else:
        data_path ="../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    if flag_data_split == "test":
        dataset_test = CustomSionnaDataset(data_path, test_split_init, test_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "val":
        dataset_test = CustomSionnaDataset(data_path, val_split_init, val_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "train":
        dataset_test = CustomSionnaDataset(data_path, train_split_init, train_split_end, flag_data_split, save_dir, indices_path)

    # Create a DataLoader 
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

    # Print the number of samples in dataloader_test
    print(f"Number of samples in {flag_data_split} dataloader: {len(dataloader_test.dataset)}")

    # Save the generated data
    os.makedirs(save_dir, exist_ok=True)

    list_times = []
    list_cosine_similarity = []
    sublist_cosine_similarity = []

    ddim.eval()
    beg_eval_time = time.time()

    # Create pandas dataframes with columns x,y and diff
    results_diff_df = pd.DataFrame(columns=['x', 'y', 'diff'])

    with torch.no_grad():
        for w in ws_test:
            ccount = 0
            pbar = tqdm(dataloader_test)
            # Iterate over the test set
            for x_test, c_test in pbar:
                x_test = np.transpose(x_test, (0, 2, 3, 1))

                x_test = x_test.to(device)
                c_test = c_test.to(device)
                c_test = c_test.detach().clone().to(device)
                sublist_cosine_similarity = [[] for _ in range(batch_size)]

                # Precompute inverse DFT codebooks (constant for this antenna config)
                Ar_inv = upa_dft_codebook(2, 2)  # shape (4, 4); inverse = Ar (since Ar.conj().T was the fwd)
                At_inv = upa_dft_codebook(8, 4)  # shape (32, 32)

                # We launch multiple sampling runs and we take the average of the cosine similarity.
                for jj in range(64):
                    start_time = time.time()
                    x_gen, _ = ddim.sample(batch_size, c_test, (2, 4, 32), device, guide_w=w)
                    end_time = time.time()

                    x_gen_ifft = x_gen

                    if jj <= 0:
                        list_times.append(end_time - start_time)
                        res = compute_upa_peak_beam_distance(
                            x_test.cpu().numpy(), x_gen_ifft.cpu().numpy(),
                            N_tx_x=8, N_tx_y=4
                        )

                        list_distances_upa.extend(res.distances.tolist())

                    # Iterate over all samples
                    for s in range(batch_size):
                        H_rt_complex = (
                            x_test.cpu().numpy()[s, 0, :, :]
                            + 1j * x_test.cpu().numpy()[s, 1, :, :]
                        )

                        H_gen_complex = (
                            x_gen_ifft.cpu().numpy()[s, 0, :, :]
                            + 1j * x_gen_ifft.cpu().numpy()[s, 1, :, :]
                        )

                        P_rt = np.sum(np.abs(H_rt_complex) ** 2, axis=0)
                        P_gen = np.sum(np.abs(H_gen_complex) ** 2, axis=0)

                        p_rt = P_rt / (np.sum(P_rt) + 1e-12)
                        p_gen = P_gen / (np.sum(P_gen) + 1e-12)

                        cos_sim = cosine_similarity(
                            p_rt.reshape(1, -1),
                            p_gen.reshape(1, -1)
                        )[0, 0]
                        sublist_cosine_similarity[s].append(cos_sim)

                for s in range(batch_size):
                    list_cosine_similarity.append(np.mean(sublist_cosine_similarity[s]))
                                
                for s in range(batch_size):
                    magnitude_ground_truth = np.sqrt(x_test.cpu().numpy()[s, 0, :, :]**2 + x_test.cpu().numpy()[s, 1, :, :]**2)
                    magnitude_ground_truth_normalized = magnitude_ground_truth / np.max(magnitude_ground_truth) # We don't need to do it since the ground truth is already normalized

                    index_max = np.unravel_index(np.argmax(magnitude_ground_truth_normalized), magnitude_ground_truth_normalized.shape)

                    magnitude_x_gen = np.sqrt(x_gen_ifft.cpu().numpy()[s, 0, :, :]**2 + x_gen_ifft.cpu().numpy()[s, 1, :, :]**2)
                    magnitude_x_gen_normalized = magnitude_x_gen / np.max(magnitude_x_gen)

                    index_max_gen = np.unravel_index(np.argmax(magnitude_x_gen_normalized), magnitude_x_gen_normalized.shape)

                    x, y, z = c_test[s, 0].cpu().numpy(), c_test[s, 1].cpu().numpy(), c_test[s, 2].cpu().numpy()
                    x = np.real(x)
                    y = np.real(y)
                    z = np.real(z)

                    peak_index_diff = np.abs((index_max[1] - index_max_gen[1]))
                    peak_index_diff = min(peak_index_diff, hist_bins - peak_index_diff) # Account for circular nature of the indices

                    new_row = {'x': x, 'y': y, 'diff': peak_index_diff}
                    results_diff_df = pd.concat([results_diff_df, pd.DataFrame([new_row])], ignore_index=True)

                    list_distances.append(peak_index_diff)
                    hist_pred[index_max_gen[1]] += 1
                    hist_gt[index_max[1]] += 1

                    if ccount % 200 == 0 and flag_data_split == "test":
                        plt.figure(figsize=(10, 5))
                        plt.subplot(2, 1, 1)
                        plt.imshow(magnitude_ground_truth_normalized, cmap='viridis')
                        plt.text(0, -2, f'Max at column {index_max[1]}', color='red', fontsize=12, ha='left', va='top')
                        plt.title(f"Label_H_test_{ccount}")
                        plt.axis('off')
                        plt.subplot(2, 1, 2)
                        plt.imshow(magnitude_x_gen_normalized, cmap='viridis')
                        plt.text(0, -2, f'Max at column {index_max_gen[1]}', color='red', fontsize=12, ha='left', va='top')
                        plt.title(f"H_test_{ccount}")
                        plt.axis('off')
                        plt.savefig(os.path.join(save_dir, f'images/H_test_{ccount}_vs_Label_H_test_{ccount}.png'), dpi=300)
                        plt.close()

                    ccount += 1

                    if ccount > num_samples_2_plot:
                        # Exit the sample generation loop
                        break
            
                print(f"Number of samples in list_cosine_similarity: {len(list_cosine_similarity)}")
                print(f"Number of elements in UPA list_distances: {len(list_distances_upa)}")
                print(f"Number of samples in list_distances: {len(list_distances)}")

                # Check if ccount is greater than num_samples_2_plot
                if ccount > num_samples_2_plot:
                    print(f"Reached the limit of {num_samples_2_plot} samples to plot. Stopping further generation.")
                    break

    print(f"Total evaluation time: {time.time() - beg_eval_time:.2f} seconds")

    if flag_data_split == "test":
        cos_sim_array = np.array(list_cosine_similarity)
        np.save(os.path.join(save_dir, 'images/list_cosine_similarity.npy'), cos_sim_array)

        fig_cos, axes_cos = plt.subplots(1, 2, figsize=(12, 4.5))

        # CDF of cosine similarity
        sns.ecdfplot(x=cos_sim_array, ax=axes_cos[0])
        axes_cos[0].set_xlabel('Cosine Similarity')
        axes_cos[0].set_ylabel('Cumulative Probability')
        axes_cos[0].set_title('CDF of Cosine Similarity')
        axes_cos[0].grid(True)

        axes_cos[1].hist(cos_sim_array, bins=50, edgecolor='black')
        axes_cos[1].set_xlabel('Cosine Similarity')
        axes_cos[1].set_ylabel('Frequency')
        axes_cos[1].set_title('Histogram of Cosine Similarity')
        axes_cos[1].grid(True)

        fig_cos.tight_layout()
        fig_cos.savefig(
            os.path.join(save_dir, 'images/cdf_and_histogram_cosine_similarity.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.close(fig_cos)

        with open(os.path.join(save_dir, 'images/cosine_similarity_stats.txt'), 'w') as f:
            f.write(f"Mean cosine similarity: {np.mean(cos_sim_array):.6f}\n")
            f.write(f"Std cosine similarity:  {np.std(cos_sim_array):.6f}\n")
            f.write(f"Min cosine similarity:  {np.min(cos_sim_array):.6f}\n")
            f.write(f"Max cosine similarity:  {np.max(cos_sim_array):.6f}\n")

        results_diff_df.to_csv(os.path.join(save_dir, 'images/results_diff.csv'), index=False)

        hist_gt_np = hist_gt.cpu().numpy()
        hist_pred_np = hist_pred.cpu().numpy()
        x_axis = np.arange(hist_bins)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        sns.ecdfplot(x=list_distances, ax=axes[0])
        axes[0].set_xlabel('Peak Index Difference')
        axes[0].set_ylabel('Cumulative Probability')
        axes[0].grid(True)

        axes[1].bar(x_axis, hist_pred_np, label='Predicted Peak Index')
        axes[1].bar(x_axis, hist_gt_np, alpha=0.5, label='Ground Truth Peak Index')
        axes[1].legend()
        axes[1].set_xlabel('Peak Index')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, 'images/cdf_and_peak_index_histogram.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.close(fig)

        np.save(os.path.join(save_dir, 'images/list_distances.npy'), np.array(list_distances))

        with open(os.path.join(save_dir, 'images/histogram_data.txt'), 'w') as f:
            f.write("Peak Index, Predicted Frequency, Ground Truth Frequency\n")
            for i in range(hist_bins):
                f.write(f"{i}, {hist_pred_np[i]}, {hist_gt_np[i]}\n")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.ecdfplot(x=list_distances_upa, ax=axes[0])
        axes[0].set_xlabel('Peak Index Difference')
        axes[0].set_ylabel('Cumulative Probability')
        axes[0].grid(True)
        axes[1].bar(x_axis, hist_pred_np, label='Predicted Peak Index')
        axes[1].bar(x_axis, hist_gt_np, alpha=0.5, label='Ground Truth Peak Index')
        axes[1].legend()
        axes[1].set_xlabel('Peak Index')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, 'images/UPA_cdf_and_peak_index_histogram.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.close(fig)
        np.save(os.path.join(save_dir, 'images/UPA_list_distances.npy'), np.array(list_distances_upa))

def main(mode):
    # python infer_cDDIM_LoS.py generate
    """
    Main function to control the flow of the script.

    Parameters:
    - mode: 'generate' to generate H_test matrices.
    """
    # Set up parameters
    batch_size = 5
    n_T = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_classes = 3
    n_feat = 256
    flag_frequency = None # It can be either 3.5 or 28 depending on which dataset we want to use
    flag_data_split = "test" # "train" or "val" or "test"
    
    save_dir = f'../../GenAI_Channel_Modeling_Models/logs/CDDIM_dataset_3.5GHz_LoS_0.0_100_28_05_2026_14_13/' # UPDATE THIS::
    ws_test = [0.0] 
    n_sample = 1

    # Check if 'images' directory exists in save_dir, if not create it
    if not os.path.exists(os.path.join(save_dir, 'images')):
        os.makedirs(os.path.join(save_dir, 'images'))

    training_config_path = os.path.join(save_dir, 'training_config.txt')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('batch_size'):
                    batch_size = int(line.split(':')[1].strip())
                elif line.startswith('n_classes'):
                    n_classes = int(line.split(':')[1].strip())
                elif line.startswith('n_feat'):
                    n_feat = int(line.split(':')[1].strip())
                elif line.startswith('n_sample'):
                    n_sample = int(line.split(':')[1].strip())
                elif line.startswith('n_T'):
                    n_T = int(line.split(':')[1].strip())
                elif line.startswith('flag_frequency'):
                    flag_frequency = float(line.split(':')[1].strip())

    ddim = DDIM(
        nn_model=ContextUnet(in_channels=2, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    )
    ddim.to(device)

    # Load the trained model
    ddim.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location=torch.device('cpu')))

    print(f"Model loaded from {os.path.join(save_dir, 'model.pth')}")

    # Print the model size
    model_size = sum(p.numel() for p in ddim.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    if mode == 'generate':
        generate_H_test(ddim, flag_frequency, flag_data_split, device, save_dir, batch_size=batch_size, n_sample=n_sample, ws_test=ws_test)
    else:
        print("Invalid mode selected. Choose 'generate'.")

if __name__ == "__main__":
    # python infer_cDDIM_LoS.py generate
    if len(sys.argv) != 2:
        print("Usage: python script.py <mode>")
        print("Modes: 'generate'")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)
