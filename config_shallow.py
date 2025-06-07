# configuration for shallow cnn on kaggle
# this file is optimized for performance on kaggle gpus.

# data settings
esc50_path = '/kaggle/input/esc-50-environmental-sound-classification'
runs_path = '/kaggle/working/results'
disable_bat_pbar = False # disable batch-level progress bar display

# cross-validation settings
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5] # use all folds for full cross-validation

# audio processing settings
sr = 44100      # sampling rate
n_mels = 128    # number of mel bins
hop_length = 512 # hop length for stft
n_fft = 1024     # fft window size

# spectrogram augmentation settings
freq_mask_param = 100
time_mask_param = 100

# model settings
model_name = 'ShallowCNN'  # 'AudioMLP', 'SimpleCNN', 'ResNet', or 'ShallowCNN'

# training settings
val_size = 0.2         # validation set size
device_id = 0
batch_size = 32        # larger batch size for cloud gpus
num_workers = 4        # more workers for faster data loading
persistent_workers = True
epochs = 30
patience = 15          # early stopping patience
lr = 1e-3
weight_decay = 2e-3

# scheduler settings
warm_epochs = 10
gamma = 0.8
step_size = 5

# testing settings
test_checkpoints = ['best_val_loss.pt'] # model to use for testing
test_experiment = '/kaggle/working/results/shallow-cnn-run'   # experiment folder for testing 