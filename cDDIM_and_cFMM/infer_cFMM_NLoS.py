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
import scipy
import matplotlib.cm as cm
from PIL import Image
import time
from scipy.io import savemat, loadmat
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from upa_beam_distance import compute_upa_peak_beam_distance

# Set the SEED for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Set CUDA_VISIBLE_DEVICES=""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
    # prepare conditioning
    cond = cond.to(device, dtype=torch.float32)
    batch_size = cond.shape[0]

    with torch.no_grad():
        x = torch.randn((batch_size, C, H, W), device=device)  
        dt = 1.0 / float(steps)

        for i in range(steps):
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
    def __init__(self, data_path_LoS, data_path_NLoS, percent_start, percent_end, flag_data_split, save_dir, indices_dir):
        self.data_path_LoS = data_path_LoS
        self.data_path_NLoS = data_path_NLoS
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
        np_array_LoS = np.load(self.data_path_LoS)
        np_array_NLoS = np.load(self.data_path_NLoS)

        data_LoS = np_array_LoS["combined_array"][:, :, 0, :, 0, 0, :] # (num_samples, num_rx, num_tx_ant, coordinates)
        data_NLoS = np_array_NLoS["combined_array"][:, :, 0, :, 0, 0, :] # (num_samples, num_rx, num_tx_ant, coordinates)

        print(f"Number of LoS samples: {data_LoS.shape[0]}")
        print(f"Number of NLoS samples: {data_NLoS.shape[0]}")
        print(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0]+data_NLoS.shape[0]}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Number of LoS samples: {data_LoS.shape[0]}\n")
            f.write(f"Number of NLoS samples: {data_NLoS.shape[0]}\n")
            f.write(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0]+data_NLoS.shape[0]}\n")

        if os.path.exists(os.path.join(self.indices_dir, 'indices_los.npy')):
            print(f"Shuffled indices found in {self.indices_dir}. Loading indices_los.npy data.")
            indices_los = np.load(os.path.join(self.indices_dir, 'indices_los.npy'))
            indices_nlos = np.load(os.path.join(self.indices_dir, 'indices_nlos.npy'))

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices loaded from {os.path.join(self.indices_dir, 'indices_los.npy and indices_nlos.npy')}\n")

            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices_los.npy')}")
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices_nlos.npy')}")
        else:
            print(f"No shuffled indices found in {self.indices_dir}. Creating shuffled indices and saving them.")
            
            indices_los = np.arange(np.shape(data_LoS)[0])
            indices_nlos = np.arange(np.shape(data_NLoS)[0])
            # Shuffle indices
            np.random.shuffle(indices_los)
            np.random.shuffle(indices_nlos)

            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices_los.npy')}")
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)
            print(f"Shuffled indices saved to {os.path.join(self.save_dir, 'indices_nlos.npy')}")

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices created and saved to {os.path.join(self.save_dir, 'indices_los.npy and indices_nlos.npy')}\n")
        
        data_LoS = data_LoS[indices_los]
        data_NLoS = data_NLoS[indices_nlos]                

        if self.percent_end > 1.0:
            total_samples = int(self.percent_end)
            n_los = total_samples // 2
            n_nlos = total_samples - n_los
            data_LoS = data_LoS[:n_los]
            data_NLoS = data_NLoS[:n_nlos]

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Using {n_los} samples from LoS and {n_nlos} samples from NLoS for a total of {total_samples} samples.\n") 

            data = np.concatenate((data_LoS, data_NLoS), axis=0)
        else:
            data = np.concatenate((data_LoS, data_NLoS), axis=0)
            self.total_length = np.shape(data)[0]

            if os.path.exists(os.path.join(self.indices_dir, 'indices.npy')):
                print(f"Shuffled indices found in {self.indices_dir}. Loading indices.npy data.")
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

            self.idx_start = int(self.percent_start * self.total_length)
            self.idx_end = int(self.percent_end * self.total_length)
            data = data[self.idx_start:self.idx_end, :, :, :] 
        
        H_set = data[:, :, :, 0]
        coords = data[:, 0, 0, 1:]
        H_set_new = H_set[:,:,:]

        array1 = np.stack((np.real(H_set_new[:,:,:]), np.imag(H_set_new[:,:,:])), axis=1)
        array2 = array1.copy()

        np.save(os.path.join(self.save_dir, self.flag_data_split+'.npy'), array1)
        print(f"** {self.flag_data_split} data saved to {os.path.join(self.save_dir, self.flag_data_split+'.npy')}")

        for i in range(array1.shape[0]) : 
            dft_data = upa_to_beamspace(array2[i,0]+1j*array2[i,1], Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i,0] = np.real(dft_data) # Real part of the DFT
            array1[i,1] = np.imag(dft_data) # Imaginary part of the DFT

        self.data = []
        self.labels = []  
        labels_array = coords

        np.save(os.path.join(self.save_dir, self.flag_data_split+'_coords.npy'), labels_array)

        for i in range(array1.shape[0]):
            magnitude = np.sqrt(array1[i, 0, :, :]**2 + array1[i, 1, :, :]**2)

            max_magnitude = np.max(magnitude)

            array1[i, 0, :, :] /= max_magnitude
            array1[i, 1, :, :] /= max_magnitude

            self.data.append(array1[i, :2, :, :])  # Appending each slice
            self.labels.append(labels_array[i])  # Appending the corresponding label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = ToTensor()(self.data[idx]).float()
        label = self.labels[idx]
        return data_item, label

def generate_H_test(model, flag_frequency, flag_data_split, steps, device, save_dir, start_idx=0, end_idx=20, batch_size=5000, n_sample=5000, ws_test=[0.0]):
    """
    Generate H_test matrices and save them to disk.

    Parameters:
    - model: The diffusion model instance.
    - steps: Number of steps for the sampling process.
    - device: The computation device ('cuda' or 'cpu').
    - save_dir: Directory to save the generated data.
    - start_idx: Starting index for data loading.
    - end_idx: Ending index for data loading.
    - batch_size: Batch size for data loading.
    - n_sample: Number of samples to generate.
    - ws_test: List of guidance weights.
    """
    list_distances = []
    list_distances_upa = []
    indices_path = save_dir

    hist_bins = 32
    hist_pred = torch.zeros(hist_bins, device=device)
    hist_gt = torch.zeros(hist_bins, device=device)

    print("** TESTING **")
    # Change the data_path below to use the 3.5GHz or the 28GHz dataset. 
    if flag_frequency == 28:
        data_path_LoS = "/raid/paulalm/datasets/6g_mirai/datasets/sina_simulations/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS.npz"
        data_path_NLoS = "/raid/paulalm/datasets/6g_mirai/datasets/sina_simulations/Final_Single_Scene_Channel_Sionna_V1_28GHz_NLoS.npz"
    else:
        data_path_LoS = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
        data_path_NLoS = "../../GenAI_Channel_Modeling_Datasets/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    if flag_data_split == "test":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, test_split_init, test_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "train":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, train_split_init, train_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "val":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, val_split_init, val_split_end, flag_data_split, save_dir, indices_path)

    # Create a DataLoader 
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

    # Print the number of samples in dataloader_test
    print(f"Number of samples in {flag_data_split} dataloader: {len(dataloader_test.dataset)}")

    # Save the generated data
    os.makedirs(save_dir, exist_ok=True)

    list_times = []
    list_cosine_similarity = []
    sublist_cosine_similarity = []

    model.eval()
    beg_eval_time = time.time()
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

                    x_gen, _ = sample_flow_matching_model(
                            model=model,
                            cond=c_test,
                            shape=(2, 4, 32),
                            device=device,
                            guide_w=w,
                            steps=steps
                        )
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

                    peak_index_diff = np.abs((index_max[1] - index_max_gen[1]))
                    peak_index_diff = min(peak_index_diff, hist_bins - peak_index_diff) # Account for circular nature of the indices
                    list_distances.append(peak_index_diff)
                    hist_pred[index_max_gen[1]] += 1
                    hist_gt[index_max[1]] += 1

                    if ccount % 500 == 0 and flag_data_split == "test":
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
                        break
            
                print(f"Number of samples in list_cosine_similarity: {len(list_cosine_similarity)}")
                print(f"Number of elements in UPA list_distances: {len(list_distances_upa)}")
                print(f"Number of samples in list_distances: {len(list_distances)}")

                if ccount > num_samples_2_plot:
                    print(f"Reached the limit of {num_samples_2_plot} samples to plot. Stopping further generation.")
                    break

    print(f"Total evaluation time: {time.time() - beg_eval_time:.2f} seconds")

    if flag_data_split == "test":
        cos_sim_array = np.array(list_cosine_similarity)
        np.save(os.path.join(save_dir, 'images/list_cosine_similarity.npy'), cos_sim_array)

        fig_cos, axes_cos = plt.subplots(1, 2, figsize=(12, 4.5))

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

        # Store mean and std of cosine similarity in a text file
        with open(os.path.join(save_dir, 'images/cosine_similarity_stats.txt'), 'w') as f:
            f.write(f"Mean cosine similarity: {np.mean(cos_sim_array):.6f}\n")
            f.write(f"Std cosine similarity:  {np.std(cos_sim_array):.6f}\n")
            f.write(f"Min cosine similarity:  {np.min(cos_sim_array):.6f}\n")
            f.write(f"Max cosine similarity:  {np.max(cos_sim_array):.6f}\n")

        # Plot CDF and peak-index histogram side by side in a single figure.
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

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(list_distances, bins=np.arange(-0.5, hist_bins+0.5, 1), kde=False)
        plt.xlabel('Peak Index Difference')
        plt.ylabel('Frequency')
        plt.title('Histogram of Peak Index Differences')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        sns.histplot(list_distances, bins=np.arange(-0.5, hist_bins+0.5, 1), kde=False, stat='percent')
        plt.xlabel('Peak Index Difference')
        plt.ylabel('Percentage of Samples')
        plt.title('Percentage of Samples for Each Peak Index Difference')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'images/peak_index_difference_histogram.png'), dpi=300)
        plt.close()

        with open(os.path.join(save_dir, 'peak_index_difference_table.txt'), 'w') as f:
            f.write("Peak Index Difference, Number of Samples, Percentage of Samples\n")
            for i in range(hist_bins):
                count = list_distances.count(i)
                percentage = (count / len(list_distances)) * 100
                f.write(f"{i}, {count}, {percentage:.2f}%\n")

        # Store list_distances in a npy file in save_dir
        np.save(os.path.join(save_dir, 'images/list_distances.npy'), np.array(list_distances))
        # Store the list_times in a npy file in save_dir
        np.save(os.path.join(save_dir, 'images/list_times.npy'), np.array(list_times))

        # Store the hist hist_pred_np and hist_gt_np values in a txt file
        with open(os.path.join(save_dir, 'images/histogram_data.txt'), 'w') as f:
            f.write("Peak Index, Predicted Frequency, Ground Truth Frequency\n")
            for i in range(hist_bins):
                f.write(f"{i}, {hist_pred_np[i]}, {hist_gt_np[i]}\n")

def main(mode):
    # python infer_cFMM_NLoS.py generate
    """
    Main function to control the flow of the script.

    Parameters:
    - mode: 'generate' to generate H_test matrices.
    """
    # Set up parameters
    batch_size = 2
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n_classes = 3
    n_feat = 256
    save_dir = "../../GenAI_Channel_Modeling_Models/logs/FMM_dataset_3.5GHz_NLoS+LoS_0.0_200_steps50/" # UPDATE THIS::
    ws_test = [0.0] 
    n_sample = 1 
    steps = 10
    flag_frequency = None # It can be either 3.5 or 28 depending on which dataset we want to use
    flag_data_split = "test" # "train" or "val" or "test"

    # Check if 'images' directory exists in save_dir, if not create it
    if not os.path.exists(os.path.join(save_dir, 'images')):
        os.makedirs(os.path.join(save_dir, 'images'))

    # Read training_config.txt from save_dir to get the parameters used during training
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
                elif line.startswith('steps'):
                    steps = int(line.split(':')[1].strip())
                elif line.startswith('flag_frequency'):
                    flag_frequency = float(line.split(':')[1].strip())

    model = ContextUnetFlowMatching(
        in_channels=2, 
        n_feat=n_feat,
        n_classes=n_classes
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location=torch.device('cpu')))

    print(f"Model loaded from {os.path.join(save_dir, 'model.pth')}")

    # Print the model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    if mode == 'generate':
        generate_H_test(model, flag_frequency, flag_data_split, steps, device, save_dir, start_idx=0, end_idx=100,
                        batch_size=batch_size, n_sample=n_sample, ws_test=ws_test)
    else:
        print("Invalid mode selected. Choose 'generate'")

if __name__ == "__main__":
    # python infer_cFMM_NLoS.py generate
    if len(sys.argv) != 2:
        print("Usage: python script.py <mode>")
        print("Modes: 'generate'")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)
