import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import torchaudio
import torchaudio.transforms as T

import config
from . import transforms

# CUDA for PyTorch is handled in the main script
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    """helper function to download a file with a progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    """downloads and extracts a zip file."""
    import zipfile
    root = os.path.dirname(file_path)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


def download_progress(current, total, width=80):
    """a simple progress bar for downloads."""
    progress_message = "downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):
    """
    a pytorch dataset for the esc-50 dataset.
    handles data loading, splitting, transformations, and augmentation.
    """
    def __init__(self, root, test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 1.0), download=True):
        root = os.path.normpath(root)

        # auto-detect audio folder
        if os.path.exists(os.path.join(root, 'audio')):
            audio_path_component = 'audio'
        elif os.path.exists(os.path.join(root, 'ESC-50-master/audio')):
            audio_path_component = 'ESC-50-master/audio'
        else:
            # if download is enabled, we assume the structure will be ESC-50-master/audio
            if download:
                audio_path_component = 'ESC-50-master/audio'
            else:
                raise RuntimeError(f"could not find esc-50 audio folder in {root}")

        audio_path = os.path.join(root, audio_path_component)
        
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        
        # download and extract the dataset if it doesn't exist
        if not os.path.exists(audio_path) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio_path
        
        # split files into train and test sets based on folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - self.test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in self.test_folds]
        
        assert set(temp) == (set(train_files) | set(test_files)) # sanity check
        
        if subset == "test":
            self.file_names = test_files
        else:
            # further split training data into train and validation sets
            if config.val_size:
                train_files, val_files = train_test_split(train_files, test_size=config.val_size, random_state=0)
            if subset == "train":
                self.file_names = train_files
            else:
                self.file_names = val_files
        
        # calculate the required length for spectrograms
        out_len = int(((config.sr * 5) // config.hop_length) * config.hop_length)
        self.n_steps = (out_len // config.hop_length) + 1
        train = self.subset == "train"

        # setup waveform transformations (padding and cropping)
        self.wave_transforms = None
        if train:
            self.wave_transforms = transforms.Compose(
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len)
            )
        else:
            self.wave_transforms = transforms.Compose(
                transforms.RandomPadding(out_len=out_len, train=False),
                transforms.RandomCrop(out_len=out_len, train=False)
            )
        
        # setup spectrogram transformations (mel spectrogram and db conversion)
        self.spec_transforms = torch.nn.Sequential(
            T.MelSpectrogram(sample_rate=config.sr, n_fft=config.n_fft, n_mels=config.n_mels, hop_length=config.hop_length),
            T.AmplitudeToDB(stype='power', top_db=80)
        )

        # setup augmentation transformations for training
        self.aug_transforms = torch.nn.Sequential()
        if train:
            self.aug_transforms.append(T.FrequencyMasking(freq_mask_param=config.freq_mask_param))
            self.aug_transforms.append(T.TimeMasking(time_mask_param=config.time_mask_param))
            
        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]
        self.n_mfcc = config.n_mfcc if hasattr(config, "n_mfcc") else None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)
        
        # load audio file using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(path, normalize=True)
        except Exception as e:
            print(f"error loading file {path}: {e}")
            # return a dummy sample on error
            return file_name, torch.zeros((1, config.n_mels, self.n_steps)), 0

        # resample if the sample rate is different from the target
        if sample_rate != config.sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=config.sr)
            waveform = resampler(waveform)

        # apply waveform transformations
        if self.wave_transforms:
             waveform = self.wave_transforms(waveform)
        
        # convert waveform to a spectrogram
        spec = self.spec_transforms(waveform)

        # apply augmentation during training
        spec = self.aug_transforms(spec)

        # normalize the spectrogram using global stats
        if self.global_mean:
            spec = (spec - self.global_mean) / self.global_std
        
        # extract the class label from the file name
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])

        return file_name, spec, class_id


def old_get_global_stats(data_path):
    """
    calculates the global mean and standard deviation for each fold of the dataset.
    this is used for normalization.
    """
    res = []
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=False)
        # concatenate all spectrograms to compute stats
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std()))
    return np.array(res)


def get_global_stats(data_path):
    """
    Calculates the global mean and standard deviation for each fold of the dataset.
    This is used for normalization.
    """

    # If path is read-only (Kaggle input), switch to working directory
    if data_path.startswith("/kaggle/input/"):
        data_path = "/kaggle/working/esc-data"
    
    # Check if audio data is already present
    audio_path1 = os.path.join(data_path, "audio")
    audio_path2 = os.path.join(data_path, "ESC-50-master/audio")
    if os.path.exists(audio_path1) or os.path.exists(audio_path2):
        download = False
    else:
        download = True

    res = []
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=download)
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std()))

    return np.array(res)
