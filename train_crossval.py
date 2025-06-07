import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
import sys
from functools import partial
import subprocess

from models.model_classifier import AudioMLP, SimpleCNN, ResNetForAudio
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50, get_global_stats
import config


# mean and std of train data for every fold
# global_stats = np.array([[-54.364834, 20.853344],
#                          [-54.279022, 20.847532],
#                          [-54.18343, 20.80387],
#                          [-54.223698, 20.798292],
#                          [-54.200905, 20.949806]])

# evaluate the model on a given dataloader
def test(model, dataloader, criterion, device):
    model.eval()

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():
        # no gradient computation needed for evaluation
        for k, x, label in tqdm(dataloader, unit='bat', disable=config.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            y_prob = model(x)
            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            # store predictions for later analysis
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch():
    # switch to training
    model.train()

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=config.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        loss = criterion(y_prob, y_true)
        optimizer.zero_grad() # reset gradients
        loss.backward() # compute gradients
        losses.append(loss.item())
        optimizer.step() # update model weights

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier():
    num_epochs = config.epochs

    # setup early stopping to prevent overfitting
    loss_stopping = EarlyStopping(patience=config.patience, delta=0.002, verbose=True, float_fmt=float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        train_acc, train_loss = train_epoch()
        val_acc, val_loss, _ = test(model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        pbar.update()
        print(end=' ')
        print(
            f"TrnAcc={train_acc:{float_fmt}}",
            f"ValAcc={val_acc:{float_fmt}}",
            f"TrnLoss={np.mean(train_loss):{float_fmt}}",
            f"ValLoss={val_loss_avg:{float_fmt}}",
            end=' ')

        early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()
        if early_stop:
            print("early stopping")
            break

        scheduler.step() # advance the learning rate scheduler

    # save the final model state
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


# build model from configuration
def make_model(n_mels, n_steps):
    model_name = config.model_name
    n_classes = config.n_classes
    print(f"Building model: {model_name}")

    if model_name == 'AudioMLP':
        model = AudioMLP(n_mels=n_mels,
                         n_steps=n_steps,
                         hidden1_size=512,
                         hidden2_size=128,
                         output_size=n_classes)
    elif model_name == 'SimpleCNN':
        model = SimpleCNN(n_classes=n_classes, n_mels=n_mels, n_steps=n_steps)
    elif model_name == 'ResNet':
        model = ResNetForAudio(n_classes=n_classes)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
    
    return model


if __name__ == "__main__":
    # prevent mac from sleeping during training
    if sys.platform == "darwin":  # macos
        caffeinate_process = subprocess.Popen(['caffeinate', '-d'])
        print("🔋 preventing mac from sleeping during training...")
    
    data_path = config.esc50_path
    
    # setup device (mps for m1 mac, cuda for nvidia, cpu for others)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.device_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    # configure logging and output directories
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    runs_path = config.runs_path
    experiment_root = os.path.join(runs_path, str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    os.makedirs(experiment_root, exist_ok=True)

    # loop over all folds for cross-validation
    scores = {}
    # calculate global mean and std for normalization across the entire dataset
    print("calculating global mean and std for normalization...")
    global_stats = get_global_stats(data_path)
    print("done.")

    for test_fold in config.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')
        if not os.path.exists(experiment):
            os.mkdir(experiment)

        # redirect stdout to a log file to keep track of training progress
        with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
                 newline='\n', proc_cr=True):
            # create a partial function for the dataset to ensure consistent settings
            get_fold_dataset = partial(ESC50, root=data_path, download=True,
                                       test_folds={test_fold}, global_mean_std=global_stats[test_fold - 1])

            train_set = get_fold_dataset(subset="train")
            print('*****')
            print(f'train folds are {train_set.train_folds} and test fold is {train_set.test_folds}')
            print('random wave cropping')

            # setup dataloader for training
            train_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       num_workers=config.num_workers,
                                                       drop_last=False,
                                                       persistent_workers=config.persistent_workers,
                                                       pin_memory=True,
                                                       )
            
            # setup dataloader for validation
            val_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="val"),
                                                     batch_size=config.batch_size,
                                                     shuffle=False,
                                                     num_workers=config.num_workers,
                                                     drop_last=False,
                                                     persistent_workers=config.persistent_workers,
                                                     )

            print()
            # instantiate model
            # get spectrogram dimensions from a sample to initialize the model correctly
            temp_dataset = get_fold_dataset(subset="train")
            _, temp_spec, _ = temp_dataset[0]
            n_mels, n_steps = temp_spec.shape[1], temp_spec.shape[2]
            
            model = make_model(n_mels=n_mels, n_steps=n_steps)
            model = model.to(device)
            print('*****')

            # define loss function and optimizer
            criterion = nn.CrossEntropyLoss().to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(),
                                         lr=config.lr,
                                         weight_decay=config.weight_decay)

            # use cosine annealing scheduler for learning rate adjustment
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=config.epochs,
                                                                  eta_min=1e-6)

            # train the model
            print()
            fit_classifier()

            # setup dataloader for testing
            test_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="test"),
                                                      batch_size=config.batch_size,
                                                      shuffle=False,
                                                      num_workers=0,  # config.num_workers,
                                                      drop_last=False,
                                                      )
            # test with best model
            # model.load_state_dict(torch.load(os.path.join(experiment, 'best_val_loss.pt')))
            model.load_state_dict(torch.load(os.path.join(experiment, 'best_val_loss.pt'), map_location=device))
            acc, _, _ = test(model, test_loader, criterion=criterion, device=device)
            scores[test_fold] = acc
            print(f'fold {test_fold} test accuracy: {acc:{float_fmt}}')
            print()
            print()

    # overall accuracy
    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['accuracy'])
    print(scores_df)
    print(f"overall accuracy: {scores_df['accuracy'].mean():{float_fmt}}")

    # Terminate the caffeinate process when the script is done
    if 'caffeinate_process' in locals() and caffeinate_process.poll() is None:
        caffeinate_process.terminate()
        print("\n✅ script finished, allowing mac to sleep again.")
