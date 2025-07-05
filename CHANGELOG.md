## 2025-07-04 – Initial EDA on ESC-50 Dataset

- Loaded `esc50.csv` metadata file from `ESC-50-master/meta/`
- Verified:
  - Dataset contains 2000 audio clips across 50 classes (40 clips per class)
  - No missing values or duplicate filenames
  - 5-fold distribution is balanced and consistent
- Visualised class distribution

## 2025-07-04 – Added SimpleResNet Model (First Improvement Step)

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
