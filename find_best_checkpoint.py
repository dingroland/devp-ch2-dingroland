import os
import torch

def find_best_checkpoint(experiment_root):
    best_val_loss = float('inf')
    best_checkpoint = None

    # Iterate over all experiment folders
    for root, dirs, files in os.walk(experiment_root):
        for file in files:
            if file.endswith('.pt'):
                checkpoint_path = os.path.join(root, file)
                
                try:
                    # this assumes the checkpoint is a dictionary with a 'val_loss' key
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    if isinstance(checkpoint, dict):
                        val_loss = checkpoint.get('val_loss', None)
                    else:
                        # if the checkpoint is just the model's state_dict, we can't get the loss
                        # we can try to get it from the file name if it's encoded there
                        # but for now, we'll just print a warning
                        print(f"warning: could not determine validation loss for {checkpoint_path}")
                        val_loss = None

                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = checkpoint_path
                except Exception as e:
                    print(f"error loading checkpoint {checkpoint_path}: {e}")

    return best_checkpoint, best_val_loss

if __name__ == "__main__":
    # adjust this path to your actual results directory
    experiment_root = r'/Users/daniel/FHTW/2_semester/project/part_2_local/devp-ch2-dingroland/result'  
    best_checkpoint, best_val_loss = find_best_checkpoint(experiment_root)
    if best_checkpoint:
        print(f"best checkpoint found at: {best_checkpoint}")
        print(f"validation loss: {best_val_loss:.4f}")
    else:
        print("no valid checkpoints found in the specified directory.") 