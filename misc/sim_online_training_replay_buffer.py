import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import from the main training file
from learn_conversation_offline_training import (
    MLModel, load_conv_data, prepare_dataloaders, calculate_metrics
)
from datasets import load_dataset


def evaluate_model_on_test(model: MLModel, test_dataloader) -> float:
    """
    Evaluate the model on test data and return the average loss.
    """
    model.classifier.eval()
    criterion = nn.CrossEntropyLoss() if model.task == "classification" else nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            try:
                input_texts, input_vals, labels = batch
            except ValueError:
                continue
                
            labels = labels.to(model._device)
            input_vals = input_vals.to(model._device).float().unsqueeze(1)
            
            if model.task == 'regression':
                labels = labels.float().unsqueeze(1)
            else:
                labels = labels.long()
            
            # Get embeddings
            embeddings = model.bert.encode(list(input_texts), convert_to_tensor=True, 
                                         batch_size=len(input_texts))
            
            # Combine features
            combined_features = torch.cat((embeddings, input_vals), dim=1)
            
            # Forward pass
            logits = model.classifier(combined_features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(input_texts)
            total_samples += len(input_texts)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def online_train_with_evaluation(
    model: MLModel, 
    train_dataloader, 
    test_dataloader,
    train_steps_per_batch: int,
    lr: float,
    buffer_size: int = 1024,
    replay_sample_size: int = 128
) -> float:
    """
    Modified version of simulate_online that returns final test loss.
    """
    import random
    from torch.utils.data import DataLoader
    
    print(f"Training with train_steps_per_batch={train_steps_per_batch}, lr={lr}")
    
    # Initialize optimizer
    model.optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    model.classifier.train()
    
    replay_buffer = []
    
    for batch_idx, batch in enumerate(train_dataloader):
        try:
            input_texts, input_vals, labels = batch
        except ValueError:
            continue

        label_name = "follow_up" if model.task == "classification" else "tta"
        
        # Add new samples to buffer
        new_samples = []
        for i in range(len(input_texts)):
            sample = {
                "text": input_texts[i],
                "turns": input_vals[i].item(),
                label_name: labels[i].item()
            }
            new_samples.append(sample)
        
        replay_buffer.extend(new_samples)
        replay_buffer = replay_buffer[-buffer_size:]  # Keep buffer size constrained

        # Perform training steps on the buffer
        if len(replay_buffer) >= replay_sample_size:
            for _ in range(train_steps_per_batch):
                # Uniform random sampling from buffer
                training_batch = random.sample(replay_buffer, replay_sample_size)
                model.train_online(training_batch, lr=lr)
    
    # Final evaluation
    final_test_loss = evaluate_model_on_test(model, test_dataloader)
    print(f"Final test loss: {final_test_loss:.4f}")
    
    return final_test_loss


def run_hyperparameter_grid_search():
    """
    Run grid search over hyperparameters and return results.
    """
    # Hyperparameter grid
    train_steps_per_batch = 1  # Fixed value
    replay_sample_size_values = [64, 128, 256, 512]
    lr_values = [1e-2, 3e-3, 1e-3, 3e-4]
    
    # Load data (using same settings as main script)
    task_type = "classification"
    dataset_choice = 'lmsys-chat-1m'
    N = -5000
    turn_equal_to = 10
    
    print("Loading dataset: LMSys-chat-1M from Hugging Face...")
    ds = load_dataset("lmsys/lmsys-chat-1m")
    df = load_conv_data(ds, N, turn_equal_to, 'lmsys')
    
    train_loader, test_loader, test_df = prepare_dataloaders(
        df=df, task=task_type, batch_size=64, test_size=0.2, random_state=42
    )
    
    # Store results
    results = []
    
    # Grid search
    total_combinations = len(replay_sample_size_values) * len(lr_values)
    current_combination = 0
    
    for replay_sample_size in replay_sample_size_values:
        for lr in lr_values:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Running combination {current_combination}/{total_combinations}")
            print(f"replay_sample_size={replay_sample_size}, lr={lr}")
            print(f"{'='*60}")
            
            # Initialize fresh model for each combination
            model = MLModel(task=task_type, dataset_choice=dataset_choice, train_mode=True)
            
            # Train and evaluate
            start_time = time.time()
            test_loss = online_train_with_evaluation(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                train_steps_per_batch=train_steps_per_batch,
                lr=lr,
                buffer_size=65536,
                replay_sample_size=replay_sample_size
            )
            training_time = time.time() - start_time
            
            # Store results
            results.append({
                'train_steps_per_batch': train_steps_per_batch,
                'replay_sample_size': replay_sample_size,
                'lr': lr,
                'test_loss': test_loss,
                'training_time': training_time
            })
            
            print(f"Completed in {training_time:.2f}s, Test Loss: {test_loss:.4f}")
    
    return pd.DataFrame(results)


def visualize_results(results_df: pd.DataFrame):
    """
    Create visualizations for the hyperparameter tuning results.
    """
    # Create pivot table for heatmap
    pivot_table = results_df.pivot(index='replay_sample_size', 
                                   columns='lr', 
                                   values='test_loss')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Heatmap of test loss
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r', 
                ax=axes[0, 0], cbar_kws={'label': 'Test Loss'})
    axes[0, 0].set_title('Test Loss Heatmap')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Replay Sample Size')
    
    # Line plot: test loss vs learning rate for different replay_sample_size
    for sample_size in results_df['replay_sample_size'].unique():
        subset = results_df[results_df['replay_sample_size'] == sample_size]
        axes[0, 1].plot(subset['lr'], subset['test_loss'], 
                       marker='o', label=f'Sample Size={sample_size}')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Test Loss vs Learning Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Line plot: test loss vs replay_sample_size for different learning rates
    for lr in results_df['lr'].unique():
        subset = results_df[results_df['lr'] == lr]
        axes[1, 0].plot(subset['replay_sample_size'], subset['test_loss'], 
                       marker='o', label=f'LR={lr:.0e}')
    axes[1, 0].set_xlabel('Replay Sample Size')
    axes[1, 0].set_ylabel('Test Loss')
    axes[1, 0].set_title('Test Loss vs Replay Sample Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: test loss vs training time
    scatter = axes[1, 1].scatter(results_df['training_time'], results_df['test_loss'], 
                                c=results_df['replay_sample_size'], s=100, 
                                cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('Test Loss')
    axes[1, 1].set_title('Test Loss vs Training Time')
    plt.colorbar(scatter, ax=axes[1, 1], label='Replay Sample Size')
    
    # Add text annotations for learning rates
    for i, row in results_df.iterrows():
        axes[1, 1].annotate(f'{row["lr"]:.0e}', 
                           (row['training_time'], row['test_loss']),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_results_table(results_df: pd.DataFrame):
    """
    Print formatted results table.
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    
    # Sort by test loss
    results_sorted = results_df.sort_values('test_loss')
    
    print(f"{'Rank':<4} {'Sample Size':<12} {'Learning Rate':<13} {'Test Loss':<10} {'Time (s)':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
        print(f"{i:<4} {row['replay_sample_size']:<12} {row['lr']:<13.0e} "
              f"{row['test_loss']:<10.4f} {row['training_time']:<8.1f}")
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    best_row = results_sorted.iloc[0]
    print(f"Train Steps Per Batch: {best_row['train_steps_per_batch']} (fixed)")
    print(f"Replay Sample Size: {best_row['replay_sample_size']}")
    print(f"Learning Rate: {best_row['lr']:.0e}")
    print(f"Test Loss: {best_row['test_loss']:.4f}")
    print(f"Training Time: {best_row['training_time']:.1f}s")
    print("="*80)


if __name__ == "__main__":
    print("Starting hyperparameter tuning for online training mode...")
    
    # Run grid search
    results_df = run_hyperparameter_grid_search()
    
    # Save results
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print(f"\nResults saved to 'hyperparameter_tuning_results.csv'")
    
    # Print results table
    print_results_table(results_df)
    
    # Create visualizations
    visualize_results(results_df)
    
    print("\nHyperparameter tuning completed!")
    print("Results visualization saved as 'hyperparameter_tuning_results.png'") 