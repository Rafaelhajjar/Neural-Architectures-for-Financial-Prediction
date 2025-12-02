"""
Evaluate advanced models and compare with baseline.
"""
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models.advanced_models import (
    DeepLateFusionNet,
    ResidualLateFusionNet,
    DeepCombinedNet
)
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.training.train_ensemble import EnsemblePredictor
from neural_nets.evaluation.metrics import compute_ranking_metrics, compute_trading_metrics


def evaluate_single_model(model, test_loader, test_df, model_name, device='cpu'):
    """Evaluate a single model."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_price = batch['X_price'].to(device)
            x_sentiment = batch['X_sentiment'].to(device)
            y = batch['y']
            
            outputs = model(x_price, x_sentiment).squeeze().cpu().numpy()
            
            all_preds.extend(outputs if isinstance(outputs, np.ndarray) else [outputs])
            all_targets.extend(y.numpy())
            all_dates.extend(batch['date'])
            all_tickers.extend(batch['ticker'])
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'date': all_dates,
        'ticker': all_tickers,
        'pred': all_preds,
        'actual_return': all_targets
    })
    
    # Compute metrics
    ranking_metrics = compute_ranking_metrics(np.array(all_targets), np.array(all_preds))
    trading_metrics, returns_df = compute_trading_metrics(pred_df, k=5)
    
    # Combine
    all_metrics = {**ranking_metrics, **trading_metrics}
    all_metrics['model'] = model_name
    
    return all_metrics, returns_df


def evaluate_ensemble(models, test_loader, test_df, model_name, device='cpu'):
    """Evaluate an ensemble of models."""
    ensemble = EnsemblePredictor(models)
    
    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_price = batch['X_price'].to(device)
            x_sentiment = batch['X_sentiment'].to(device)
            y = batch['y']
            
            # Ensemble prediction
            outputs = ensemble(x_price, x_sentiment).squeeze().cpu().numpy()
            
            all_preds.extend(outputs if isinstance(outputs, np.ndarray) else [outputs])
            all_targets.extend(y.numpy())
            all_dates.extend(batch['date'])
            all_tickers.extend(batch['ticker'])
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'date': all_dates,
        'ticker': all_tickers,
        'pred': all_preds,
        'actual_return': all_targets
    })
    
    # Compute metrics
    ranking_metrics = compute_ranking_metrics(np.array(all_targets), np.array(all_preds))
    trading_metrics, returns_df = compute_trading_metrics(pred_df, k=5)
    
    # Combine
    all_metrics = {**ranking_metrics, **trading_metrics}
    all_metrics['model'] = model_name
    
    return all_metrics, returns_df


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("EVALUATING ADVANCED MODELS")
    print("="*70)
    print(f"Device: {device}\n")
    
    # Load data
    _, _, test_df = load_and_prepare_data()
    
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    loaders = create_data_loaders(
        test_df, test_df, test_df,
        price_features, sentiment_features,
        task='regression',
        batch_size=256
    )
    
    results = []
    
    # ========================================================================
    # Evaluate Advanced Models
    # ========================================================================
    
    # 1. Deep Late Fusion
    print("1. Evaluating Deep Late Fusion...")
    try:
        model1 = DeepLateFusionNet(task='regression')
        checkpoint = torch.load('neural_nets/trained_models/deep_late_fusion_mse_best.pt', map_location=device)
        model1.load_state_dict(checkpoint['model_state_dict'])
        metrics1, _ = evaluate_single_model(model1, loaders['test'], test_df, 'Deep Late Fusion', device)
        results.append(metrics1)
        print(f"   Sharpe: {metrics1['sharpe_ratio']:.4f} | Return: {metrics1['total_return']*100:.1f}%\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # 2. Residual Late Fusion
    print("2. Evaluating Residual Late Fusion...")
    try:
        model2 = ResidualLateFusionNet(task='regression')
        checkpoint = torch.load('neural_nets/trained_models/residual_late_fusion_mse_best.pt', map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        metrics2, _ = evaluate_single_model(model2, loaders['test'], test_df, 'Residual Late Fusion', device)
        results.append(metrics2)
        print(f"   Sharpe: {metrics2['sharpe_ratio']:.4f} | Return: {metrics2['total_return']*100:.1f}%\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # 3. Deep Combined
    print("3. Evaluating Deep Combined...")
    try:
        model3 = DeepCombinedNet(task='regression')
        checkpoint = torch.load('neural_nets/trained_models/deep_combined_mse_best.pt', map_location=device)
        model3.load_state_dict(checkpoint['model_state_dict'])
        metrics3, _ = evaluate_single_model(model3, loaders['test'], test_df, 'Deep Combined', device)
        results.append(metrics3)
        print(f"   Sharpe: {metrics3['sharpe_ratio']:.4f} | Return: {metrics3['total_return']*100:.1f}%\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # 4. Deep Late Fusion Ensemble
    print("4. Evaluating Deep Late Fusion Ensemble...")
    try:
        # Load all ensemble members
        ensemble_models = []
        for seed in [42, 43, 44, 45, 46]:
            model = DeepLateFusionNet(task='regression')
            checkpoint = torch.load(f'neural_nets/trained_models/deep_late_fusion_ensemble_seed{seed}_best.pt',
                                   map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            ensemble_models.append(model)
        
        metrics4, _ = evaluate_ensemble(ensemble_models, loaders['test'], test_df, 
                                       'Deep Late Fusion Ensemble (5)', device)
        results.append(metrics4)
        print(f"   Sharpe: {metrics4['sharpe_ratio']:.4f} | Return: {metrics4['total_return']*100:.1f}%\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # ========================================================================
    # Load Baseline Results for Comparison
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING BASELINE RESULTS FOR COMPARISON")
    print("="*70)
    
    baseline_df = pd.read_csv('neural_nets/results/ranking_results.csv')
    
    # ========================================================================
    # Create Comparison Table
    # ========================================================================
    print("\n" + "="*70)
    print("COMPLETE MODEL COMPARISON")
    print("="*70)
    print("\nBASELINE MODELS:")
    print(baseline_df[['model', 'spearman', 'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']].to_string(index=False))
    
    if results:
        print("\n\nADVANCED MODELS:")
        advanced_df = pd.DataFrame(results)
        print(advanced_df[['model', 'spearman', 'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']].to_string(index=False))
        
        # Save advanced results
        advanced_df.to_csv('neural_nets/results/advanced_models_results.csv', index=False)
        print(f"\n✅ Advanced results saved to: neural_nets/results/advanced_models_results.csv")
        
        # Combined results
        all_results = pd.concat([baseline_df, advanced_df], ignore_index=True)
        all_results.to_csv('neural_nets/results/all_models_results.csv', index=False)
        print(f"✅ Combined results saved to: neural_nets/results/all_models_results.csv")
        
        # Find best
        print("\n" + "="*70)
        print("🏆 BEST MODELS")
        print("="*70)
        
        best_overall = all_results.loc[all_results['sharpe_ratio'].idxmax()]
        print(f"\nBest Overall (Sharpe):")
        print(f"  Model: {best_overall['model']}")
        print(f"  Sharpe: {best_overall['sharpe_ratio']:.4f}")
        print(f"  Return: {best_overall['total_return']*100:.1f}%")
        print(f"  $10,000 → ${10000 * (1 + best_overall['total_return']):.0f}")
        
        # Check if advanced beat baseline
        baseline_best_sharpe = baseline_df['sharpe_ratio'].max()
        advanced_best_sharpe = advanced_df['sharpe_ratio'].max() if len(advanced_df) > 0 else 0
        
        print(f"\nBaseline best Sharpe: {baseline_best_sharpe:.4f}")
        print(f"Advanced best Sharpe: {advanced_best_sharpe:.4f}")
        
        if advanced_best_sharpe > baseline_best_sharpe:
            improvement = (advanced_best_sharpe - baseline_best_sharpe) / baseline_best_sharpe * 100
            print(f"✅ Improvement: +{improvement:.1f}%")
        else:
            print("⚠️ Advanced models did not beat baseline")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nNext: Regenerate plots with new models included!")


if __name__ == "__main__":
    main()

