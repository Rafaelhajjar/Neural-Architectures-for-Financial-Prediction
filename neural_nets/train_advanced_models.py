"""
Train all advanced model architectures.

Models to train:
1. Deep Late Fusion (6 layers + BatchNorm)
2. Residual Late Fusion (Residual blocks + BatchNorm)
3. Deep Combined (6 layers early fusion + BatchNorm)
4. Deep Late Fusion Ensemble (5 members with different seeds)

This will take approximately 2-3 hours total.
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models.advanced_models import (
    DeepLateFusionNet,
    ResidualLateFusionNet,
    DeepCombinedNet
)
from neural_nets.training.train_ranker import train_ranker
from neural_nets.training.train_ensemble import train_ensemble


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("TRAINING ADVANCED MODELS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Models to train: 4 (3 single + 1 ensemble)")
    print("="*70)
    
    num_epochs = 100
    batch_size = 256
    learning_rate = 0.001
    
    results = []
    
    # ========================================================================
    # MODEL 1: Deep Late Fusion (Single)
    # ========================================================================
    print("\n\n" + "="*70)
    print("MODEL 1/4: Deep Late Fusion (6 layers + BatchNorm)")
    print("="*70)
    
    try:
        model1, hist1 = train_ranker(
            model_class=DeepLateFusionNet,
            model_name='deep_late_fusion_mse',
            loss_type='mse',
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        results.append(('Deep Late Fusion', 'SUCCESS', hist1['val_loss'][-1]))
        print("\n✅ Deep Late Fusion trained successfully!")
    except Exception as e:
        print(f"\n❌ Deep Late Fusion failed: {e}")
        results.append(('Deep Late Fusion', 'FAILED', None))
    
    # ========================================================================
    # MODEL 2: Residual Late Fusion (Single)
    # ========================================================================
    print("\n\n" + "="*70)
    print("MODEL 2/4: Residual Late Fusion (Residual blocks + BatchNorm)")
    print("="*70)
    
    try:
        model2, hist2 = train_ranker(
            model_class=ResidualLateFusionNet,
            model_name='residual_late_fusion_mse',
            loss_type='mse',
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        results.append(('Residual Late Fusion', 'SUCCESS', hist2['val_loss'][-1]))
        print("\n✅ Residual Late Fusion trained successfully!")
    except Exception as e:
        print(f"\n❌ Residual Late Fusion failed: {e}")
        results.append(('Residual Late Fusion', 'FAILED', None))
    
    # ========================================================================
    # MODEL 3: Deep Combined (Single)
    # ========================================================================
    print("\n\n" + "="*70)
    print("MODEL 3/4: Deep Combined (6 layers early fusion + BatchNorm)")
    print("="*70)
    
    try:
        model3, hist3 = train_ranker(
            model_class=DeepCombinedNet,
            model_name='deep_combined_mse',
            loss_type='mse',
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        results.append(('Deep Combined', 'SUCCESS', hist3['val_loss'][-1]))
        print("\n✅ Deep Combined trained successfully!")
    except Exception as e:
        print(f"\n❌ Deep Combined failed: {e}")
        results.append(('Deep Combined', 'FAILED', None))
    
    # ========================================================================
    # MODEL 4: Deep Late Fusion Ensemble (5 members)
    # ========================================================================
    print("\n\n" + "="*70)
    print("MODEL 4/4: Deep Late Fusion Ensemble (5 members)")
    print("This will take longer (~5x single model time)")
    print("="*70)
    
    try:
        models_ens, hists_ens = train_ensemble(
            model_class=DeepLateFusionNet,
            base_name='deep_late_fusion_ensemble',
            num_members=5,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Average validation loss across ensemble members
        avg_val_loss = sum(h['val_loss'][-1] for h in hists_ens) / len(hists_ens)
        results.append(('Deep Late Fusion Ensemble', 'SUCCESS', avg_val_loss))
        print("\n✅ Ensemble trained successfully!")
        
        # Save ensemble metadata
        import json
        ensemble_info = {
            'num_members': 5,
            'seeds': [42, 43, 44, 45, 46],
            'base_model': 'DeepLateFusionNet',
            'val_losses': [h['val_loss'][-1] for h in hists_ens],
            'avg_val_loss': avg_val_loss
        }
        
        with open('neural_nets/trained_models/deep_late_fusion_ensemble_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print(f"\nEnsemble info saved to trained_models/deep_late_fusion_ensemble_info.json")
        
    except Exception as e:
        print(f"\n❌ Ensemble failed: {e}")
        results.append(('Deep Late Fusion Ensemble', 'FAILED', None))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    for name, status, val_loss in results:
        loss_str = f"Val Loss: {val_loss:.6f}" if val_loss else ""
        print(f"{name:30s} {status:10s} {loss_str}")
    
    successful = sum(1 for _, status, _ in results if status == 'SUCCESS')
    print(f"\nSuccessful: {successful}/{len(results)}")
    
    if successful == len(results):
        print("\n🎉 All advanced models trained successfully!")
        print("\nNext steps:")
        print("  1. Run evaluation: python neural_nets/evaluate_advanced_models.py")
        print("  2. Compare with baseline models")
        print("  3. Generate updated plots")
    else:
        print("\n⚠️ Some models failed to train. Check errors above.")
    
    print("="*70)


if __name__ == "__main__":
    main()

