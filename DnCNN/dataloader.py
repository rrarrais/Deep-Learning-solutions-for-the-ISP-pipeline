import os
import warnings
from posixpath import split

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, utils

# Ignore warnings
warnings.filterwarnings("ignore")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input, output = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        output = output.transpose((2, 0, 1))
        return (torch.from_numpy(input),
                torch.from_numpy(output))


# Zurich dataset class
class ZurichDataset(Dataset):
    """
    Zurich RAW-to-DLSR dataset class

    This class expects a root folder containing the following structure:
    -root_dir
        -train
            -canon
            -huawei_raw
        -test
            -canon
            -huawei_raw
    """

    def __init__(self,
                 root_dir:str,
                 input_src:str="huawei_raw",
                 output_src:str="canon",
                 train: bool = True,
                 transform=None):
        """
        Args:
            root_dir (string): Directory with expected dataset structure.
            train (bool): Boolean representing train or test split usage.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.input_src = input_src
        self.output_src = output_src
        self.train = train
        self.check_structure_and_load()

    def check_structure_and_load(self):
        # Check root_dir
        assert (os.path.isdir(self.root_dir)), f"Path {self.root_dir} does not exist"

        # Check train/test folder
        middle = "train" if self.train else "test"
        middle_dir = os.path.join(self.root_dir, middle)
        assert (os.path.isdir(middle_dir)), f"Missing {middle} folder inside {self.root_dir}"

        # Check input folder
        self.input_dir = os.path.join(middle_dir, self.input_src)
        assert (os.path.isdir(self.input_dir)), f"Missing {self.input_src} folder inside {middle_dir}"

        # Check output folder
        self.output_dir = os.path.join(middle_dir, self.output_src)
        assert (os.path.isdir(self.output_dir)), f"Missing {self.output_src} folder inside {middle_dir}"

        # Check if file list inside input and output folders match
        input_list = os.listdir(self.input_dir)
        input_extension = input_list[0].split(".")[1]
        input_list = sorted([int(file.split(".")[0]) for file in input_list])
        output_list = os.listdir(self.output_dir)
        output_extension = output_list[0].split(".")[1]
        output_list = sorted([int(file.split(".")[0]) for file in output_list])
        assert (input_list == output_list), f"Input and output folders have different item lists."

        # Save info items
        self.length = len(input_list)
        self.image_names = input_list
        self.input_extension = input_extension
        self.output_extension = output_extension

    def extract_bayer_channels(self, raw):
        # Reshape the input bayer image
        ch_B  = raw[1::2, 1::2]
        ch_Gb = raw[0::2, 1::2]
        ch_R  = raw[0::2, 0::2]
        ch_Gr = raw[1::2, 0::2]

        RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
        RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

        return RAW_norm


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Input image name formulation and load
        input_name = os.path.join(self.input_dir,
                                  f"{self.image_names[idx]}.{self.input_extension}")
        input = np.asarray(io.imread(input_name))
        input = self.extract_bayer_channels(input)

        # Output image name formulation and load
        output_name = os.path.join(self.output_dir,
                                  f"{self.image_names[idx]}.{self.output_extension}")
        output = np.asarray(io.imread(output_name))
        output = output.astype(np.float32) / 255

        # Sample construction
        sample = (input, output)

        if self.transform:
            sample = self.transform(sample)

        return sample
