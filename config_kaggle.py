# Kaggle-optimized configuration for maximum performance
# dir with ESC50 data
esc50_path = '/kaggle/working/esc50'  # Kaggle working dir is writeable

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]  # Train ALL folds for maximum performance

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
n_fft = 1024

# Aggressive spectrogram augmentation for Kaggle
freq_mask_param = 100  # Increased from 80
time_mask_param = 100  # Increased from 80

model_name = 'ResNet'  # Use ResNet for best performance

# ###TRAINING
# ratio to split off from training data
val_size = .2
device_id = 0
batch_size = 32  # Larger batch size for cloud GPUs
num_workers = 4  # More workers for faster data loading
persistent_workers = True
epochs = 100  # More epochs for better convergence
# early stopping after epochs with no improvement
patience = 15  # Less patience for faster convergence
lr = 1e-3
weight_decay = 3e-3  # Higher regularization
warm_epochs = 10
gamma = 0.8
step_size = 5

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['best_val_loss.pt']
# experiment folder will be auto-generated
test_experiment = 'results/kaggle-run' 