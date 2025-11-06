# File: model_evaluation.py
# Purpose: Evaluate and compare all anomaly detection models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, df, true_label_col='is_anomaly'):
        self.df = df
        self.true_label_col = true_label_col
        self.y_true = df[true_label_col].values
        self.results = {}
        
    def calculate_metrics(self, y_pred, method_name):
        
        metrics = {
            'method': method_name,
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
            'true_positives': np.sum((self.y_true == 1) & (y_pred == 1)),
            'false_positives': np.sum((self.y_true == 0) & (y_pred == 1)),
            'true_negatives': np.sum((self.y_true == 0) & (y_pred == 0)),
            'false_negatives': np.sum((self.y_true == 1) & (y_pred == 0)),
            'total_predicted_anomalies': np.sum(y_pred),
            'total_actual_anomalies': np.sum(self.y_true)
        }
        
        return metrics
    
    def evaluate_all_methods(self):
        """Evaluate all prediction methods in the dataframe"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Find all prediction columns
        pred_cols = [col for col in self.df.columns if col.startswith('pred_')]
        
        results_list = []
        
        for pred_col in pred_cols:
            method_name = pred_col.replace('pred_', '')
            y_pred = self.df[pred_col].values
            
            metrics = self.calculate_metrics(y_pred, method_name)
            results_list.append(metrics)
            self.results[method_name] = metrics
            
        # Create results dataframe
        results_df = pd.DataFrame(results_list)
        
        # Sort by F1 score
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 80)
        print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Detected':<12}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            print(f"{row['method']:<25} {row['precision']:<12.4f} {row['recall']:<12.4f} "
                  f"{row['f1_score']:<12.4f} {int(row['total_predicted_anomalies']):<12}")
        
        print("-" * 80)
        print(f"Total actual anomalies: {int(self.y_true.sum())}")
        print("="*80)
        
        return results_df
    
    def plot_confusion_matrices(self, methods=None, save_path='plots/confusion_matrices.png'):
        
        import os
        os.makedirs('plots', exist_ok=True)
        
        if methods is None:
            pred_cols = [col for col in self.df.columns if col.startswith('pred_')]
            methods = [col.replace('pred_', '') for col in pred_cols]
        
        n_methods = len(methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, method in enumerate(methods):
            y_pred = self.df[f'pred_{method}'].values
            cm = confusion_matrix(self.y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            axes[idx].set_title(f'{method}\nF1: {self.results[method]["f1_score"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(len(methods), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, score_cols=None, save_path='plots/roc_curves.png'):
        
        import os
        os.makedirs('plots', exist_ok=True)
        
        if score_cols is None:
            # Find error/score columns
            score_cols = [col for col in self.df.columns if 'error_' in col or 'score_' in col]
        
        if not score_cols:
            print("No score columns found for ROC curves")
            return
        
        plt.figure(figsize=(10, 8))
        
        for score_col in score_cols:
            method_name = score_col.replace('error_', '').replace('score_', '')
            scores = self.df[score_col].values
            
            # For errors, higher is more anomalous
            # For some scores, lower might be more anomalous - adjust as needed
            if 'error_' in score_col:
                y_scores = scores
            else:
                y_scores = -scores  # Invert if needed
            
            fpr, tpr, _ = roc_curve(self.y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{method_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, score_cols=None, save_path='plots/pr_curves.png'):
        
        import os
        os.makedirs('plots', exist_ok=True)
        
        if score_cols is None:
            score_cols = [col for col in self.df.columns if 'error_' in col or 'score_' in col]
        
        if not score_cols:
            print("No score columns found for PR curves")
            return
        
        plt.figure(figsize=(10, 8))
        
        for score_col in score_cols:
            method_name = score_col.replace('error_', '').replace('score_', '')
            scores = self.df[score_col].values
            
            if 'error_' in score_col:
                y_scores = scores
            else:
                y_scores = -scores
            
            precision, recall, _ = precision_recall_curve(self.y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{method_name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")
        plt.close()
    
    def plot_anomaly_detection_timeline(self, methods=None, sample_size=1000, 
                                       save_path='plots/anomaly_timeline.png'):
        
        import os
        os.makedirs('plots', exist_ok=True)
        
        if methods is None:
            pred_cols = [col for col in self.df.columns if col.startswith('pred_')]
            methods = [col.replace('pred_', '') for col in pred_cols][:3]  # Limit to 3
        
        # Sample data for visualization
        if len(self.df) > sample_size:
            df_sample = self.df.iloc[:sample_size].copy()
        else:
            df_sample = self.df.copy()
        
        # Get first sensor for visualization
        sensor_cols = [col for col in df_sample.columns if col.startswith('sensor_')]
        if not sensor_cols:
            print("No sensor columns found for timeline plot")
            return
        
        sensor = sensor_cols[0]
        
        fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(15, 3*(len(methods)+1)))
        
        # Plot original data with true anomalies
        axes[0].plot(df_sample.index, df_sample[sensor], linewidth=0.5, alpha=0.7, color='blue')
        true_anomalies = df_sample[df_sample[self.true_label_col] == 1]
        axes[0].scatter(true_anomalies.index, true_anomalies[sensor], 
                       color='red', s=30, alpha=0.7, label='True Anomalies')
        axes[0].set_title(f'{sensor} - True Anomalies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot predictions for each method
        for idx, method in enumerate(methods, 1):
            axes[idx].plot(df_sample.index, df_sample[sensor], linewidth=0.5, alpha=0.7, color='blue')
            pred_anomalies = df_sample[df_sample[f'pred_{method}'] == 1]
            axes[idx].scatter(pred_anomalies.index, pred_anomalies[sensor],
                            color='orange', s=30, alpha=0.7, label=f'Detected by {method}')
            axes[idx].set_title(f'{sensor} - {method} Predictions')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.xlabel('Sample Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly timeline saved to {save_path}")
        plt.close()
    
    def generate_comparison_report(self, save_path='evaluation_report.txt'):
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANOMALY DETECTION MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Total Actual Anomalies: {int(self.y_true.sum())} "
                   f"({self.y_true.sum()/len(self.y_true)*100:.2f}%)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            for method, metrics in sorted(self.results.items(), 
                                         key=lambda x: x[1]['f1_score'], 
                                         reverse=True):
                f.write(f"\n{method.upper()}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  True Negatives: {metrics['true_negatives']}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']}\n")
                f.write(f"  Total Detected: {metrics['total_predicted_anomalies']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            
            # Find best method
            best_method = max(self.results.items(), key=lambda x: x[1]['f1_score'])
            f.write(f"\nBest Overall Performance: {best_method[0]}\n")
            f.write(f"  F1-Score: {best_method[1]['f1_score']:.4f}\n")
            
            # Find method with highest recall
            best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
            f.write(f"\nHighest Recall: {best_recall[0]}\n")
            f.write(f"  Recall: {best_recall[1]['recall']:.4f}\n")
            
            # Find method with highest precision
            best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
            f.write(f"\nHighest Precision: {best_precision[0]}\n")
            f.write(f"  Precision: {best_precision[1]['precision']:.4f}\n")
            
        print(f"\nComparison report saved to {save_path}")
    
    def run_full_evaluation(self):
        print("\n" + "="*80)
        print("RUNNING FULL EVALUATION")
        print("="*80)
        
        # Calculate metrics
        results_df = self.evaluate_all_methods()
        
        # Generate all plots
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_anomaly_detection_timeline()
        
        # Generate report
        self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return results_df

if __name__ == "__main__":
    # Combine results from statistical and deep learning methods
    df_stat = pd.read_csv('results_statistical.csv')
    df_dl = pd.read_csv('results_deep_learning.csv')
    
    # Merge predictions
    dl_pred_cols = [col for col in df_dl.columns if col.startswith('pred_') or col.startswith('error_')]
    for col in dl_pred_cols:
        df_stat[col] = df_dl[col]
    
    # Run evaluation
    evaluator = ModelEvaluator(df_stat, true_label_col='is_anomaly')
    results = evaluator.run_full_evaluation()
    
    print("\nTop 3 Methods by F1-Score:")
    print(results[['method', 'precision', 'recall', 'f1_score']].head(3))