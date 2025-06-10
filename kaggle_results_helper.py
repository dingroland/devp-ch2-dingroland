#!/usr/bin/env python3
"""
Kaggle Results Helper - Access and prepare training results for download
"""
import os
import pandas as pd
import shutil
import zipfile
from pathlib import Path

def check_kaggle_results():
    """Check what results are available in Kaggle"""
    
    results_paths = [
        '/kaggle/working/results',
        '/kaggle/working',
        'results',
        '.'
    ]
    
    print("🔍 CHECKING FOR RESULTS IN KAGGLE...")
    print("=" * 50)
    
    for path in results_paths:
        if os.path.exists(path):
            print(f"\n📁 Found directory: {path}")
            try:
                contents = os.listdir(path)
                if contents:
                    print(f"   Contents: {contents}")
                    
                    # Look for timestamped experiment directories
                    exp_dirs = [d for d in contents if os.path.isdir(os.path.join(path, d)) 
                               and any(c.isdigit() for c in d)]
                    if exp_dirs:
                        print(f"   🎯 Experiment directories: {exp_dirs}")
                        return path, exp_dirs
                else:
                    print("   (empty)")
            except Exception as e:
                print(f"   Error reading: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    return None, []

def summarize_kaggle_experiment(results_path, exp_dir):
    """Summarize a specific experiment in Kaggle"""
    
    exp_path = os.path.join(results_path, exp_dir)
    print(f"\n📊 EXPERIMENT: {exp_dir}")
    print("-" * 40)
    
    if not os.path.exists(exp_path):
        print("❌ Experiment directory not found!")
        return None
    
    # Get fold directories
    fold_dirs = [d for d in os.listdir(exp_path) 
                 if os.path.isdir(os.path.join(exp_path, d)) and d.isdigit()]
    
    if not fold_dirs:
        print("❌ No fold directories found!")
        return None
    
    print(f"✅ Found {len(fold_dirs)} folds: {sorted(fold_dirs)}")
    
    # Collect results
    results = []
    for fold in sorted(fold_dirs):
        fold_path = os.path.join(exp_path, fold)
        
        # Check files
        files = {
            'best_model': os.path.exists(os.path.join(fold_path, 'best_val_loss.pt')),
            'terminal_model': os.path.exists(os.path.join(fold_path, 'terminal.pt')),
            'train_log': os.path.exists(os.path.join(fold_path, 'train.log')),
            'test_scores': os.path.exists(os.path.join(fold_path, 'test_scores.csv'))
        }
        
        print(f"   Fold {fold}: {sum(files.values())}/4 files present")
        
        # Read test scores if available
        scores_file = os.path.join(fold_path, 'test_scores.csv')
        if os.path.exists(scores_file):
            try:
                df = pd.read_csv(scores_file)
                acc = df[df['metric'] == 'TestAcc']['value'].iloc[0]
                loss = df[df['metric'] == 'TestLoss']['value'].iloc[0]
                results.append({
                    'fold': int(fold), 
                    'test_accuracy': acc, 
                    'test_loss': loss
                })
                print(f"      Test Accuracy: {acc:.3f}")
            except Exception as e:
                print(f"      ⚠️  Error reading scores: {e}")
    
    if results:
        df = pd.DataFrame(results)
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Mean Accuracy: {df['test_accuracy'].mean():.3f} ± {df['test_accuracy'].std():.3f}")
        print(f"   Best Fold: {df.loc[df['test_accuracy'].idxmax(), 'fold']} ({df['test_accuracy'].max():.3f})")
        return df
    
    return None

def prepare_kaggle_submission(results_path, exp_dir, output_name="kaggle_submission"):
    """Prepare results for download from Kaggle"""
    
    exp_path = os.path.join(results_path, exp_dir)
    
    # Create a clean submission directory
    submission_dir = f"/kaggle/working/{output_name}"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    print(f"\n📦 PREPARING SUBMISSION: {output_name}")
    print("-" * 40)
    
    # Get fold directories
    fold_dirs = [d for d in os.listdir(exp_path) 
                 if os.path.isdir(os.path.join(exp_path, d)) and d.isdigit()]
    
    # Collect all results into summary
    all_results = []
    for fold in sorted(fold_dirs):
        fold_path = os.path.join(exp_path, fold)
        scores_file = os.path.join(fold_path, 'test_scores.csv')
        
        if os.path.exists(scores_file):
            try:
                df = pd.read_csv(scores_file)
                acc = df[df['metric'] == 'TestAcc']['value'].iloc[0]
                loss = df[df['metric'] == 'TestLoss']['value'].iloc[0]
                all_results.append({
                    'fold': int(fold),
                    'test_accuracy': acc,
                    'test_loss': loss
                })
            except:
                pass
    
    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(submission_dir, 'cross_validation_results.csv'), index=False)
        
        # Create summary statistics
        summary = {
            'metric': ['mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy'],
            'value': [
                results_df['test_accuracy'].mean(),
                results_df['test_accuracy'].std(),
                results_df['test_accuracy'].min(),
                results_df['test_accuracy'].max()
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(submission_dir, 'summary_statistics.csv'), index=False)
        print("✅ Results summaries saved")
    
    # Copy best models (optional - these are large files)
    models_dir = os.path.join(submission_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for fold in sorted(fold_dirs):
        fold_path = os.path.join(exp_path, fold)
        best_model = os.path.join(fold_path, 'best_val_loss.pt')
        if os.path.exists(best_model):
            # Only copy if file size is reasonable
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            if size_mb < 100:  # Less than 100MB
                shutil.copy2(best_model, os.path.join(models_dir, f'fold_{fold}_best.pt'))
                print(f"✅ Model for fold {fold} copied ({size_mb:.1f}MB)")
            else:
                print(f"⚠️  Model for fold {fold} too large ({size_mb:.1f}MB) - skipped")
    
    # Copy training logs
    logs_dir = os.path.join(submission_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    for fold in sorted(fold_dirs):
        fold_path = os.path.join(exp_path, fold)
        log_file = os.path.join(fold_path, 'train.log')
        if os.path.exists(log_file):
            shutil.copy2(log_file, os.path.join(logs_dir, f'fold_{fold}_train.log'))
    
    print("✅ Training logs copied")
    
    # Create a zip file for easy download
    zip_path = f"/kaggle/working/{output_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, submission_dir)
                zipf.write(file_path, arcname)
    
    print(f"📁 Submission package created: {zip_path}")
    print(f"📁 Submission directory: {submission_dir}")
    
    return submission_dir, zip_path

def main():
    """Main function to check and prepare results"""
    
    print("🚀 KAGGLE RESULTS HELPER")
    print("=" * 50)
    
    # Check for results
    results_path, exp_dirs = check_kaggle_results()
    
    if not results_path or not exp_dirs:
        print("\n❌ NO RESULTS FOUND!")
        print("\nTo generate results, run:")
        print("   python train_kaggle.py")
        return
    
    print(f"\n✅ Found results in: {results_path}")
    
    # Process each experiment
    best_exp = None
    best_acc = 0
    
    for exp_dir in exp_dirs:
        results_df = summarize_kaggle_experiment(results_path, exp_dir)
        if results_df is not None:
            mean_acc = results_df['test_accuracy'].mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_exp = exp_dir
    
    if best_exp:
        print(f"\n🏆 BEST EXPERIMENT: {best_exp} (Mean Acc: {best_acc:.3f})")
        
        # Prepare submission
        submission_dir, zip_path = prepare_kaggle_submission(results_path, best_exp)
        
        print(f"\n🎯 SUBMISSION READY!")
        print(f"📥 Download this file: {zip_path}")
        print(f"📁 Or browse directory: {submission_dir}")
    else:
        print("\n⚠️  No complete experiments found!")

if __name__ == "__main__":
    main() 