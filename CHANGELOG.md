## 2025-07-04 – Initial EDA on ESC-50 Dataset

- Loaded `esc50.csv` metadata file from `ESC-50-master/meta/`
- Verified:
  - Dataset contains 2000 audio clips across 50 classes (40 clips per class)
  - No missing values or duplicate filenames
  - 5-fold distribution is balanced and consistent
- Visualised class distribution

## 2025-07-04 – Added SimpleResNet Model

- Implemented a minimal ResNet-style CNN architecture (`SimpleResNet`) to improve from the baseline MLP
- Given the sequential and structured nature of log-Mel spectrograms (time × frequency), a ResNet was chosen
- Architecture includes:
  - Initial Conv2D layer with ReLU and BatchNorm
  - A single ResidualBlock (2 × Conv2D + skip connection)
  - Global average pooling followed by a linear classifier
- Inputs: log-Mel spectrograms of shape (1, n_mels, n_steps)
- Outputs: class logits of shape (batch_size, 50)
- Training setup:
  - 5-fold cross-validation
  - SGD optimizer with StepLR and weight decay
  - Early stopping on validation loss
- Data Augmentation:
  - Reused the baseline augmentation strategy


## 2025-07-04 – Improvement to ResNet

- Extended the baseline `SimpleResNet` architecture to use 4 residual blocks, the first model with only 1 underfitted.
- Each block maintains 16 channels, no downsampling.
- Training on 100 epochs again
- training acc at around 16% after 40 epochs -> still underfitting

## 2025-07-05 – Deeper ResNet

- Downsampling and channel expansion (16 → 32 → 64)
- Global avg pooling + dropout before final classifier
- Same input/output shape and training setup as before
- Still uses baseline data augmentation (random crop + pad)
- training acc at around 22% after 40 epochs -> still underfitting

## 2025-07-05 – Learning Rate Scheduling Added

- Switched from fixed `StepLR` to adaptive `ReduceLROnPlateau` scheduler
- Reduces LR by factor of 0.5 if validation loss doesn't improve for 5 epochs
- Increased initial learning rate to `1e-2` for faster early convergence
- LR now printed per epoch for better training insight
- data augmentation fixed to return minimum signal length
- 60% acc after first fold shows strong improvement. still not enough training stopped.


## 2025-07-05 – Switched to OneCycleLR Scheduler

- Replaced `ReduceLROnPlateau` with `OneCycleLR` for more dynamic learning rate adaptation
- Uses cosine annealing with warm-up:
  - `max_lr = 1e-2`, `div_factor = 25`, `final_div_factor = 1e4`
  - Warm-up over first 25% of epochs, then gradual cosine decay
- Scheduler now steps per batch instead of per epoch
- Added a fourth residual block to the `ResNet` backbone:
  - `BetterResidualBlock(64 → 64)` added after the third block
- No additional downsampling to preserve final spatial resolution
- Increases model capacity while retaining feature map size
- Enabled GPU acceleration with automatic detection of available CUDA devices
- Activated `torch.nn.DataParallel` for multi-GPU training (2× NVIDIA T4 on Kaggle)
- time per epoch reduced for 30s to 15s
- TestAcc    0.670 , TestLoss   1.220 after first fold 

## 2025-07-05 – Add more Augmentation
- model seems to underfit data -> increase `warm_up` to 25% of epochs
- TestAcc    0.660, TestLoss   1.238 after one fold. still not performing good enough
- Add `RandomScale`, `RandomNoise`, `FrequencyMask` and `TimeMask`

## 2025-07-06 – Model, Optimizer 
- Increased model capacity to better capture complex features and combat underfitting.
- Deepened the `ResNet` by adding a fifth residual block.
- Widened the network by increasing channels in the third stage 
- Replaced `SGD` with `AdamW` for potentially faster convergence and improved generalisation
- Adjusted max learning rate to `1e-3`


## 2025-07-07 – Disable parallel Data Loading
- Training froze after 1.5 folds most likely most likely due to worker freeze 
- Set `num_workers` to `0` to prevent the training process from freezing in the Kaggle environment
- Now running at around 45s/ epoch 


## 2025-07-07 – Added Mixed Precision 
- Integrated `torch.cuda.amp` with `autocast` and `GradScaler` to better use the T4 GPU's Tensor Cores
- By performing most operations in `float16` instead of `float32` a speedup in epoch time should be visible
- `GradScaler` is instantiated only once per training run 
- The `autocast` context manager is used to automatically convert model operations to use the faster `float16` data type
- A `GradScaler` was implemented to dynamically scale the loss. This prevents gradients from being rounded to zero in `float16` by keeping them in a representable range
- Training at around 40s/ epoch still too slow

## 2025-07-07 – Spawn instead of fork
- The training process can  hang and deadlock when using multiple `DataLoader` workers (`num_workers > 0`)
- The default multiprocessing start method on Linux, `fork`, is unsafe. It copies the parent process's state but not its threads, which can lead to deadlocks when a worker process inherits a locked resource without the thread to unlock it
- Set the start method to `spawn`. This creates fresh, independent worker processes that do not inherit the parent's state to avoid a deadlock
- Slight startup overhead of `spawn` is reduced by using `persistent_workers=True`
- Mean Test Acc 0.795, Test Loss 0.746 this should be enough for the 72% threshold