# data settings
esc50_path = 'data/esc50'
runs_path = 'results'
disable_bat_pbar = False # disable batch-level progress bar display

# cross-validation settings
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]

# audio processing settings
sr = 44100      # sampling rate
n_mels = 128    # number of mel bins
hop_length = 512 # hop length for stft
n_fft = 1024     # fft window size

# spectrogram augmentation settings
freq_mask_param = 80
time_mask_param = 80

# model settings
model_name = 'ShallowCNN'  # 'AudioMLP', 'SimpleCNN', 'ResNet', or 'ShallowCNN'

# training settings
val_size = 0.2         # validation set size
device_id = 0
batch_size = 16
num_workers = 0        # set to 0 for local execution, especially on windows
persistent_workers = False
epochs = 50
patience = 15          # early stopping patience
lr = 1e-3
weight_decay = 2e-3

# scheduler settings
warm_epochs = 10
gamma = 0.8
step_size = 5

# testing settings
test_checkpoints = ['best_val_loss.pt'] # model to use for testing
test_experiment = 'results/sample-run'   # experiment folder for testing 