import os
import sys
import time
import pandas as pd
import subprocess
from typing import List, Dict, Any
import json
import numpy as np

# Add the current directory to Python path to import the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main script functions
from ml_model import (
    load_conv_data, 
    load_conv_data_with_assistant, 
    prepare_dataloaders,
    BaselineModel, 
    BaselineModel2,
    MLModel,
    calculate_metrics,
    matthews_corrcoef
)

def run_single_experiment(
    dataset_choice: str,
    combined_dataset: bool,
    max_context_turn: int,
    N: int = 20000,
    turn_equal_to: int = 20,
    task_type: str = "classification",
    test_size: float = 10000,
    batch_size: int = 256,
    num_epochs: int = 20,
    lr: float = 5e-5
) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters and return metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_choice}, max_context_turn={max_context_turn}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if True:
        # Load individual dataset (always without assistant messages)
        if dataset_choice == "lmsys-chat-1m":
            print("Loading dataset: LMSys-chat-1M from Hugging Face...")
            from datasets import load_dataset
            ds = load_dataset("lmsys/lmsys-chat-1m")
            df = load_conv_data(ds, N, turn_equal_to, 'lmsys')
        elif dataset_choice == "sharegpt":
            print(f"Loading dataset: ShareGPT")
            df = load_conv_data("../../../ShareGPT_V3_unfiltered_cleaned_split.json", N, turn_equal_to, 'sharegpt')
        elif dataset_choice == "tay":
            print(f"Loading dataset: Tay")
            df = load_conv_data("../../tay.json", N, turn_equal_to, 'tay')
        elif dataset_choice == "chatbot_arena":
            print("Loading dataset: Chatbot Arena Conversations from Hugging Face...")
            from datasets import load_dataset
            ds = load_dataset("lmsys/chatbot_arena_conversations")
            df = load_conv_data(ds, N, turn_equal_to, 'chatbot_arena')
        elif dataset_choice == "gpt4":
            print("Loading dataset: GPT4 Conversations from Hugging Face...")
            from datasets import load_dataset
            ds = load_dataset("lightblue/gpt4_conversations_multilingual")
            df = load_conv_data(ds, N, turn_equal_to, 'gpt4')
        else:
            raise ValueError(f"Unknown dataset: {dataset_choice}")
    
    # Prepare dataloaders
    train_loader, test_loader, train_df, test_df = prepare_dataloaders(
        df=df, task=task_type, batch_size=batch_size, test_size=test_size
    )
    
    # Run baselines
    print("\n--- Running Baselines ---")
    
    # Baseline 1
    baseline = BaselineModel(train_df)
    baseline.train()
    y_true = test_df['follow_up'].values
    y_pred = baseline.predict(len(test_df))
    mcc_baseline = matthews_corrcoef(y_true, y_pred)
    metrics_baseline = calculate_metrics(y_true, y_pred)
    
    # Baseline 2
    baseline2 = BaselineModel2(train_df)
    baseline2.train()
    y_pred2 = test_df['turns'].apply(baseline2.predict).values
    mcc_baseline2 = matthews_corrcoef(y_true, y_pred2)
    metrics_baseline2 = calculate_metrics(y_true, y_pred2)
    
    print(f"Baseline1 - F1 micro: {metrics_baseline['f1 micro']:.4f}, F1 macro: {metrics_baseline['f1 macro']:.4f}, MCC: {mcc_baseline:.4f}")
    print(f"Baseline2 - F1 micro: {metrics_baseline2['f1 micro']:.4f}, F1 macro: {metrics_baseline2['f1 macro']:.4f}, MCC: {mcc_baseline2:.4f}")
    
    # Initialize and train ML model
    print("\n--- Training ML Model ---")
    dataset_name = "max_context_turns" if max_context_turn else dataset_choice
    mlmodel = MLModel(task=task_type, dataset_choice=dataset_name)
    
    # Load best model and evaluate
    print("\n--- Loading Best Model and Evaluating ---")
    try:
        if dataset_choice == "sharegpt":
            if max_context_turn == 5:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_metric_0_2767_epoch_16.pt"
            elif max_context_turn == 10:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_metric_0_3000_epoch_11.pt"
        elif dataset_choice == "lmsys-chat-1m":
            if max_context_turn == 5:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_metric_0_3609_epoch_18.pt"
            elif max_context_turn == 10:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_metric_0_3545_epoch_4.pt"
        elif dataset_choice == "chatbot_arena":
            if max_context_turn == 5:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_metric_0_2939_epoch_10.pt"
            elif max_context_turn == 10:
                best_checkpoint_path = "/home/ubuntu/nuerips8753/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_metric_0_2252_epoch_6.pt"
        
        print(f"Loading checkpoint: {best_checkpoint_path}")
        mlmodel_loaded = MLModel(task=task_type, dataset_choice=dataset_name)
        mlmodel_loaded.load_model(best_checkpoint_path)
        
        # Calculate best threshold from train_loader (similar to get_predictor_accuracy.py)
        print("\n--- Calculating Best Threshold from Training Data ---")
        train_true = []
        train_probs = []
        
        for batch in train_loader:
            batch_texts, batch_turns, batch_labels = batch
            batch_texts = list(batch_texts)
            batch_turns = batch_turns.tolist()
            batch_labels = batch_labels.tolist()
            batch_probs = mlmodel_loaded.predict_batch_processed(batch_texts, batch_turns)
            train_true.extend(batch_labels)
            train_probs.extend(batch_probs)
        
        # Calculate best threshold using precision_recall_curve
        from sklearn.metrics import precision_recall_curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(train_true, train_probs)
        
        # Best threshold by max MCC
        mccs = []
        for t in thresholds:
            y_pred_t = [1 if prob > t else 0 for prob in train_probs]
            mccs.append(matthews_corrcoef(train_true, y_pred_t))
        
        mccs = np.array(mccs)
        best_mcc_idx = mccs.argmax()
        best_mcc_threshold = thresholds[best_mcc_idx] if best_mcc_idx < len(thresholds) else 0.5
        max_mcc_train = mccs[best_mcc_idx]
        print(f"Best threshold (MCC) from training data: {best_mcc_threshold:.4f} for max MCC: {max_mcc_train:.4f}")
        
        # Evaluate on test set using the best threshold
        print("\n--- Evaluating on Test Set ---")
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
        
        # Final predictions with best threshold from training data
        y_pred_final = [1 if prob > best_mcc_threshold else 0 for prob in all_probs]
        final_metrics = calculate_metrics(all_true, y_pred_final)
        max_mcc = matthews_corrcoef(all_true, y_pred_final)
        
        print(f"ML Model - F1 micro: {final_metrics['f1 micro']:.4f}, F1 macro: {final_metrics['f1 macro']:.4f}, MCC: {max_mcc:.4f}")
        
        experiment_time = time.time() - start_time
        
        return {
            'dataset': dataset_choice,
            'max_context_turn': max_context_turn,
            'N': N,
            'turn_equal_to': turn_equal_to,
            'num_epochs': num_epochs,
            'lr': lr,
            'experiment_time_seconds': experiment_time,
            'best_threshold': best_mcc_threshold,
            'baseline1_f1_micro': metrics_baseline['f1 micro'],
            'baseline1_f1_macro': metrics_baseline['f1 macro'],
            'baseline1_mcc': mcc_baseline,
            'baseline2_f1_micro': metrics_baseline2['f1 micro'],
            'baseline2_f1_macro': metrics_baseline2['f1 macro'],
            'baseline2_mcc': mcc_baseline2,
            'ml_model_f1_micro': final_metrics['f1 micro'],
            'ml_model_f1_macro': final_metrics['f1 macro'],
            'ml_model_mcc': max_mcc,
            'ml_model_precision': final_metrics['precision'],
            'ml_model_recall': final_metrics['recall']
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            'dataset': dataset_choice,
            'max_context_turn': max_context_turn,
            'N': N,
            'turn_equal_to': turn_equal_to,
            'num_epochs': num_epochs,
            'lr': lr,
            'experiment_time_seconds': time.time() - start_time,
            'error': str(e)
        }

def run_all_experiments(
    datasets: List[str] = None,
    turn_list: List[bool] = None,
    output_csv: str = "max_context_turn_experiment_results.csv"
):
    """
    Run experiments with different datasets and combined dataset configurations.
    """
    if datasets is None:
        datasets = ["chatbot_arena", "lmsys-chat-1m", "sharegpt", "tay", "gpt4"]
    
    if turn_list is None:
        turn_list = [5, 10]
    
    results = []
    
    for dataset in datasets:
        for max_context_turn in turn_list:
            try:
                result = run_single_experiment(
                    dataset_choice=dataset,
                    combined_dataset=False,
                    max_context_turn=max_context_turn,
                    N=20000,
                    turn_equal_to=20,
                    task_type="classification",
                    test_size=10000,
                    batch_size=256,
                    num_epochs=20,
                    lr=5e-5
                )
                results.append(result)
                
                # Save intermediate results
                df_intermediate = pd.DataFrame(results)
                df_intermediate.to_csv(output_csv, index=False)
                print(f"Intermediate results saved to {output_csv}")
                
            except Exception as e:
                print(f"Error in experiment {dataset}, max_context_turn={max_context_turn}: {e}")
                results.append({
                    'dataset': dataset,
                    'max_context_turn': max_context_turn,
                    'error': str(e)
                })
    
    # Save final results
    df_final = pd.DataFrame(results)
    df_final.to_csv(output_csv, index=False)
    print(f"\nFinal results saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if 'error' not in r]
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful_results)}")
    print(f"Failed experiments: {len(results) - len(successful_results)}")
    
    if successful_results:
        print("\nBest ML Model Results by F1 Macro:")
        df_successful = pd.DataFrame(successful_results)
        best_f1_macro = df_successful.loc[df_successful['ml_model_f1_macro'].idxmax()]
        print(f"Dataset: {best_f1_macro['dataset']}, Combined Dataset: {best_f1_macro['max_context_turn']}")
        print(f"F1 Macro: {best_f1_macro['ml_model_f1_macro']:.4f}, MCC: {best_f1_macro['ml_model_mcc']:.4f}")
        
        print("\nBest ML Model Results by MCC:")
        best_mcc = df_successful.loc[df_successful['ml_model_mcc'].idxmax()]
        print(f"Dataset: {best_mcc['dataset']}, Combined Dataset: {best_mcc['max_context_turn']}")
        print(f"F1 Macro: {best_mcc['ml_model_f1_macro']:.4f}, MCC: {best_mcc['ml_model_mcc']:.4f}")
        
        # Compare combined vs individual datasets
        print("\nCombined vs Individual Dataset Comparison:")
        combined_results = df_successful[df_successful['max_context_turn'] == True]
        individual_results = df_successful[df_successful['max_context_turn'] == False]
        
        if not combined_results.empty and not individual_results.empty:
            avg_combined_f1 = combined_results['ml_model_f1_macro'].mean()
            avg_individual_f1 = individual_results['ml_model_f1_macro'].mean()
            avg_combined_mcc = combined_results['ml_model_mcc'].mean()
            avg_individual_mcc = individual_results['ml_model_mcc'].mean()
            
            print(f"Average F1 Macro - Combined: {avg_combined_f1:.4f}, Individual: {avg_individual_f1:.4f}")
            print(f"Average MCC - Combined: {avg_combined_mcc:.4f}, Individual: {avg_individual_mcc:.4f}")
    
    return df_final

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experiments with different datasets and combined dataset configurations')
    parser.add_argument('--datasets', nargs='+', default=["chatbot_arena", "lmsys-chat-1m", "sharegpt"],
                       help='List of datasets to run experiments on')
    parser.add_argument('--max_context_turn', nargs='+', type=int, default=[5, 10],
                       help='List of boolean values for max_context_turn')
    parser.add_argument('--output_csv', default='max_context_turn_experiment_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--single_experiment', action='store_true',
                       help='Run only a single experiment with default parameters')
    
    args = parser.parse_args()
    
    if args.single_experiment:
        # Run just one experiment for testing
        result = run_single_experiment(
            dataset_choice="chatbot_arena",
            max_context_turn=5
        )
        print(f"Single experiment result: {result}")
    else:
        # Run all experiments
        run_all_experiments(
            datasets=args.datasets,
            turn_list=args.max_context_turn,
            output_csv=args.output_csv
        ) 