import json
import os
import pandas as pd
import torch
import torch.nn as nn
import time
import sys
sys.setrecursionlimit(15000)
import math
from torch.optim import Adam
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve

# Import the necessary functions from the original file
from ml_model import (
    load_conv_data,
    prepare_dataloaders,
    MLModel,
    BaselineModel,
    BaselineModel2,
    calculate_metrics
)

def load_combined_datasets(N_per_dataset=50000, turn_equal_to=20):
    """
    Load data from three datasets: sharegpt, chatbot_arena, and lmsys-chat-1m
    Each dataset will contribute N_per_dataset samples
    """
    print(f"Loading {N_per_dataset} samples from each of the three datasets...")
    
    all_dataframes = []
    
    # 1. Load ShareGPT dataset
    print("\n--- Loading ShareGPT dataset ---")
    try:
        sharegpt_df = load_conv_data(
            "../../../ShareGPT_V3_unfiltered_cleaned_split.json", 
            N_per_dataset, 
            turn_equal_to, 
            'sharegpt'
        )
        print(f"ShareGPT: Loaded {len(sharegpt_df)} samples")
        all_dataframes.append(sharegpt_df)
    except Exception as e:
        print(f"Error loading ShareGPT: {e}")
    
    # 2. Load Chatbot Arena dataset
    print("\n--- Loading Chatbot Arena dataset ---")
    try:
        ds_chatbot = load_dataset("lmsys/chatbot_arena_conversations")
        chatbot_df = load_conv_data(
            ds_chatbot, 
            N_per_dataset, 
            turn_equal_to, 
            'chatbot_arena'
        )
        print(f"Chatbot Arena: Loaded {len(chatbot_df)} samples")
        all_dataframes.append(chatbot_df)
    except Exception as e:
        print(f"Error loading Chatbot Arena: {e}")
    
    # 3. Load LMSys Chat-1M dataset
    print("\n--- Loading LMSys Chat-1M dataset ---")
    try:
        ds_lmsys = load_dataset("lmsys/lmsys-chat-1m")
        lmsys_df = load_conv_data(
            ds_lmsys, 
            N_per_dataset, 
            turn_equal_to, 
            'lmsys'
        )
        print(f"LMSys Chat-1M: Loaded {len(lmsys_df)} samples")
        all_dataframes.append(lmsys_df)
    except Exception as e:
        print(f"Error loading LMSys Chat-1M: {e}")
    
    # Combine all dataframes
    if not all_dataframes:
        raise ValueError("No datasets were successfully loaded!")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n--- Combined Dataset Summary ---")
    print(f"Total samples: {len(combined_df)}")
    print(f"Follow-up distribution: {combined_df['follow_up'].value_counts().to_dict()}")
    print(f"Turns distribution: {combined_df['turns'].value_counts().sort_index().to_dict()}")
    
    return combined_df

def main():
    # Configuration
    task_type = "classification"
    N_per_dataset = 30000  # 50k samples from each dataset
    turn_equal_to = 20
    num_epochs = 20
    learning_rate = 5e-5
    batch_size = 256
    test_size = 0.02
    
    print("=== Training Combined Model ===")
    print(f"Configuration:")
    print(f"- Samples per dataset: {N_per_dataset}")
    print(f"- Turn limit: {turn_equal_to}")
    print(f"- Task type: {task_type}")
    print(f"- Batch size: {batch_size}")
    print(f"- Test size: {test_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {num_epochs}")
    
    # Load combined dataset
    combined_df = load_combined_datasets(N_per_dataset, turn_equal_to)
    
    # Prepare dataloaders
    print("\n--- Preparing DataLoaders ---")
    train_loader, test_loader, train_df, test_df = prepare_dataloaders(
        df=combined_df, 
        task=task_type, 
        batch_size=batch_size, 
        test_size=test_size
    )
    
    # Run baseline models
    print("\n--- Running Baseline Models ---")
    
    # Baseline 1: Overall follow-up ratio
    baseline = BaselineModel(train_df)
    baseline.train()
    y_true = test_df['follow_up'].values
    y_pred = baseline.predict(len(test_df))
    mcc_baseline = matthews_corrcoef(y_true, y_pred)
    print(f"BaselineModel MCC: {mcc_baseline:.4f}")
    metrics = calculate_metrics(y_true, y_pred)
    print(f"BaselineModel F1 micro: {metrics['f1 micro']:.4f}, F1 macro: {metrics['f1 macro']:.4f}")
    
    # Baseline 2: Based on conversation turns
    baseline2 = BaselineModel2(train_df)
    baseline2.train()
    y_pred2 = test_df['turns'].apply(baseline2.predict).values
    mcc_baseline2 = matthews_corrcoef(y_true, y_pred2)
    print(f"BaselineModel2 MCC: {mcc_baseline2:.4f}")
    metrics2 = calculate_metrics(y_true, y_pred2)
    print(f"BaselineModel2 F1 micro: {metrics2['f1 micro']:.4f}, F1 macro: {metrics2['f1 macro']:.4f}")
    
    # Initialize and train ML model
    print("\n--- Training ML Model ---")
    mlmodel = MLModel(
        task=task_type, 
        dataset_choice='combined_datasets',
        bert_model_name="intfloat/multilingual-e5-small",
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    )
    
    # Create checkpoint directory
    checkpoint_dir = f"/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_combined_{turn_equal_to}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the model
    mlmodel.train(
        train_dataloader=train_loader,
        train_df=train_df,
        test_df=test_df,
        num_epochs=num_epochs,
        lr=learning_rate,
        save_dir=checkpoint_dir
    )
    
    # Load best model and evaluate
    print("\n--- Loading Best Model and Final Evaluation ---")
    try:
        saved_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        saved_files.sort(reverse=False)  # Sort by metric value (ascending for MCC)
        best_checkpoint_path = os.path.join(checkpoint_dir, saved_files[0])
        print(f"Loading best checkpoint: {best_checkpoint_path}")
        
        # Load the model
        mlmodel_loaded = MLModel(task=task_type, dataset_choice='combined_datasets')
        mlmodel_loaded.load_model(best_checkpoint_path)
        
        # Evaluate on test set
        print("\n--- Final Test Set Evaluation ---")
        all_true = []
        all_probs = []
        
        for batch in test_loader:
            batch_texts, batch_turns, batch_labels = batch
            batch_texts = list(batch_texts)
            batch_turns = batch_turns.tolist()
            batch_labels = batch_labels.tolist()
            
            batch_probs = mlmodel_loaded.predict_batch_processed(batch_texts, batch_turns)
            all_true.extend(batch_labels)
            all_probs.extend(batch_probs)
        
        # Calculate best threshold and final metrics
        precision_curve, recall_curve, thresholds = precision_recall_curve(all_true, all_probs)
        
        # Find best threshold by MCC
        mccs = []
        for t in thresholds:
            y_pred_t = [1 if prob > t else 0 for prob in all_probs]
            mccs.append(matthews_corrcoef(all_true, y_pred_t))
        
        mccs = np.array(mccs)
        best_mcc_idx = mccs.argmax()
        best_mcc_threshold = thresholds[best_mcc_idx] if best_mcc_idx < len(thresholds) else 0.5
        max_mcc = mccs[best_mcc_idx]
        
        # Final predictions with best threshold
        y_pred_final = [1 if prob > best_mcc_threshold else 0 for prob in all_probs]
        
        # Calculate final metrics
        final_mcc = matthews_corrcoef(all_true, y_pred_final)
        final_precision = precision_score(all_true, y_pred_final, zero_division=0)
        final_recall = recall_score(all_true, y_pred_final, zero_division=0)
        final_f1_micro = f1_score(all_true, y_pred_final, average='micro', zero_division=0)
        final_f1_macro = f1_score(all_true, y_pred_final, average='macro', zero_division=0)
        
        print(f"\n=== Final Results ===")
        print(f"Best threshold (MCC): {best_mcc_threshold:.4f}")
        print(f"Final MCC: {final_mcc:.4f}")
        print(f"Final Precision: {final_precision:.4f}")
        print(f"Final Recall: {final_recall:.4f}")
        print(f"Final F1 Micro: {final_f1_micro:.4f}")
        print(f"Final F1 Macro: {final_f1_macro:.4f}")
        
        # Save final results
        results = {
            'best_threshold': best_mcc_threshold,
            'mcc': final_mcc,
            'precision': final_precision,
            'recall': final_recall,
            'f1_micro': final_f1_micro,
            'f1_macro': final_f1_macro,
            'total_samples': len(combined_df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'configuration': {
                'N_per_dataset': N_per_dataset,
                'turn_equal_to': turn_equal_to,
                'task_type': task_type,
                'batch_size': batch_size,
                'test_size': test_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs
            }
        }
        
        results_file = os.path.join(checkpoint_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    except IndexError:
        print(f"\nNo checkpoints found in '{checkpoint_dir}'.")
    except Exception as e:
        print(f"\nError during final evaluation: {e}")

if __name__ == "__main__":
    main() 