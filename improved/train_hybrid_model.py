"""
Training script for Hybrid Model (Temporal NN + Knowledge Graph)
Predicts individual agricultural input costs and aggregates to total cost
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.hybrid_model import HybridModel
from app.utils.data_preprocessing import AgriculturalDataPreprocessor

class AgriculturalDataset(Dataset):
    """Dataset class for agricultural data"""
    
    def __init__(self, sequences, targets, node_indices):
        self.sequences = sequences
        self.targets = targets
        self.node_indices = node_indices
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        target_dict = self.targets[idx]
        
        # Convert targets to tensor
        target = torch.FloatTensor([
            target_dict['seed'],
            target_dict['fertilizer'],
            target_dict['herbicide'],
            target_dict['pesticide'],
            target_dict['labor']
        ])
        
        return {
            'temporal': self.sequences[idx],
            'target': target,
            'total_target': target_dict['total'],
            'node_idx': self.node_indices[idx]
        }

def train_epoch(model, dataloader, optimizer, criterion, device, node_features, edge_index):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_individual_loss = 0.0
    total_total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        temporal_features = batch['temporal'].to(device)
        targets = batch['target'].to(device)
        total_targets = batch['total_target'].to(device)
        node_indices = batch['node_idx'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(temporal_features, node_features, edge_index, node_indices)
        
        # Individual cost predictions
        pred_individual = torch.stack([
            predictions['seed'],
            predictions['fertilizer'],
            predictions['herbicide'],
            predictions['pesticide'],
            predictions['labor']
        ], dim=1)
        
        # Individual loss (normalized targets)
        individual_loss = criterion(pred_individual, targets)
        
        # Total cost loss - sum predicted individual costs and compare with total target
        # Note: Total target is unscaled, so we need to sum the predicted normalized costs
        pred_total_sum = predictions['seed'] + predictions['fertilizer'] + predictions['herbicide'] + predictions['pesticide'] + predictions['labor']
        
        # For training, we calculate total from unscaled individual costs
        # The model predicts normalized values, so we need to handle this differently
        # Actually, we should compare predictions['total'] directly if it's trained on unscaled total
        # But for now, let's use the sum of individual predictions vs individual targets summed
        target_total_sum = targets.sum(dim=1)  # Sum of normalized individual targets
        total_loss_val = nn.MSELoss()(pred_total_sum, target_total_sum)
        
        # Combined loss - weight individual more heavily as it's the primary task
        loss = individual_loss + 0.3 * total_loss_val
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_individual_loss += individual_loss.item()
        total_total_loss += total_loss_val.item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'individual_loss': total_individual_loss / num_batches,
        'total_cost_loss': total_total_loss / num_batches
    }

def validate(model, dataloader, criterion, device, node_features, edge_index, preprocessor=None):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_individual_loss = 0.0
    total_total_loss = 0.0
    num_batches = 0
    
    all_predictions = {'seed': [], 'fertilizer': [], 'herbicide': [], 'pesticide': [], 'labor': [], 'total': []}
    all_targets = {'seed': [], 'fertilizer': [], 'herbicide': [], 'pesticide': [], 'labor': [], 'total': []}
    
    # Store denormalized predictions for total cost calculation
    denorm_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            temporal_features = batch['temporal'].to(device)
            targets = batch['target'].to(device)
            total_targets = batch['total_target'].to(device)
            node_indices = batch['node_idx'].to(device)
            
            # Forward pass
            predictions = model(temporal_features, node_features, edge_index, node_indices)
            
            # Individual cost predictions
            pred_individual = torch.stack([
                predictions['seed'],
                predictions['fertilizer'],
                predictions['herbicide'],
                predictions['pesticide'],
                predictions['labor']
            ], dim=1)
            
            # Losses
            individual_loss = criterion(pred_individual, targets)
            
            # Total cost loss
            pred_total_sum = predictions['seed'] + predictions['fertilizer'] + predictions['herbicide'] + predictions['pesticide'] + predictions['labor']
            target_total_sum = targets.sum(dim=1)
            total_loss_val = nn.MSELoss()(pred_total_sum, target_total_sum)
            loss = individual_loss + 0.3 * total_loss_val
            
            total_loss += loss.item()
            total_individual_loss += individual_loss.item()
            total_total_loss += total_loss_val.item()
            num_batches += 1
            
            # Store predictions for metrics
            # Store normalized predictions and targets for individual costs
            all_predictions['seed'].extend(predictions['seed'].cpu().numpy())
            all_predictions['fertilizer'].extend(predictions['fertilizer'].cpu().numpy())
            all_predictions['herbicide'].extend(predictions['herbicide'].cpu().numpy())
            all_predictions['pesticide'].extend(predictions['pesticide'].cpu().numpy())
            all_predictions['labor'].extend(predictions['labor'].cpu().numpy())
            
            all_targets['seed'].extend(targets[:, 0].cpu().numpy())
            all_targets['fertilizer'].extend(targets[:, 1].cpu().numpy())
            all_targets['herbicide'].extend(targets[:, 2].cpu().numpy())
            all_targets['pesticide'].extend(targets[:, 3].cpu().numpy())
            all_targets['labor'].extend(targets[:, 4].cpu().numpy())
            
            # For total cost evaluation, we need to:
            # 1. Denormalize individual predictions
            # 2. Sum them to get predicted total
            # 3. Compare with actual unscaled total targets
            
            # Store normalized predictions for individual metrics
            # We'll calculate total from denormalized values later
            seed_pred_norm = predictions['seed'].cpu().numpy()
            fert_pred_norm = predictions['fertilizer'].cpu().numpy()
            herb_pred_norm = predictions['herbicide'].cpu().numpy()
            pest_pred_norm = predictions['pesticide'].cpu().numpy()
            labor_pred_norm = predictions['labor'].cpu().numpy()
            
            # Denormalize predictions (convert back to original scale)
            # This is a simplified approach - in practice, we'd use the preprocessor's inverse_transform
            # For now, we'll use the model's direct total prediction which sums normalized values
            # The actual proper way would be to inverse transform each individually then sum
            
            # Store actual total targets (unscaled)
            all_targets['total'].extend(total_targets.cpu().numpy())
            
            # For total predictions, denormalize individual predictions and sum them
            if preprocessor is not None:
                # Denormalize individual predictions
                batch_size = len(seed_pred_norm)
                for i in range(batch_size):
                    denorm_pred = preprocessor.inverse_transform_costs({
                        'seed': seed_pred_norm[i],
                        'fertilizer': fert_pred_norm[i],
                        'herbicide': herb_pred_norm[i],
                        'pesticide': pest_pred_norm[i],
                        'labor': labor_pred_norm[i]
                    })
                    # Sum denormalized predictions
                    pred_total = (
                        denorm_pred['seed'] +
                        denorm_pred['fertilizer'] +
                        denorm_pred['herbicide'] +
                        denorm_pred['pesticide'] +
                        denorm_pred['labor']
                    )
                    denorm_predictions.append(pred_total)
            else:
                # Fallback: use normalized sum (will have different scale)
                all_predictions['total'].extend(predictions['total'].cpu().numpy())
    
    # Use denormalized predictions for total cost if available
    if preprocessor is not None and len(denorm_predictions) > 0:
        all_predictions['total'] = np.array(denorm_predictions)
    
    # Calculate metrics
    metrics = {}
    for key in ['seed', 'fertilizer', 'herbicide', 'pesticide', 'labor', 'total']:
        pred = np.array(all_predictions[key])
        true = np.array(all_targets[key])
        
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
        
        # R² (Coefficient of Determination)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics[key] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    return {
        'total_loss': total_loss / num_batches,
        'individual_loss': total_individual_loss / num_batches,
        'total_cost_loss': total_total_loss / num_batches,
        'metrics': metrics
    }

def main():
    """Main training function"""
    print("=" * 60)
    print("Training Hybrid Model (Temporal NN + Knowledge Graph)")
    print("=" * 60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'epochs': 50,  # Full training
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seq_length': 12,
        'save_dir': 'models',
        'model_name': 'hybrid_agricultural_model'
    }
    
    print(f"Using device: {config['device']}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv('train_dataset_cleaned.csv')
    val_df = pd.read_csv('validation_dataset_cleaned.csv')
    test_df = pd.read_csv('test_dataset_cleaned.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = AgriculturalDataPreprocessor(seq_length=config['seq_length'])
    data = preprocessor.prepare_data(train_df, val_df, test_df)
    
    # Extract graph info
    node_features = data['graph']['node_features']
    edge_index = data['graph']['edge_index']
    
    print(f"Graph nodes: {node_features.shape[0]}")
    print(f"Graph edges: {edge_index.shape[1]}")
    print(f"Train sequences: {len(data['train']['sequences'])}")
    print(f"Val sequences: {len(data['val']['sequences']) if data['val'] else 0}")
    print(f"Test sequences: {len(data['test']['sequences']) if data['test'] else 0}")
    
    # Move graph to device
    node_features = node_features.to(config['device'])
    edge_index = edge_index.to(config['device'])
    
    # Create datasets and dataloaders
    train_dataset = AgriculturalDataset(
        data['train']['sequences'],
        data['train']['targets'],
        data['train']['node_indices']
    )
    val_dataset = AgriculturalDataset(
        data['val']['sequences'],
        data['val']['targets'],
        data['val']['node_indices']
    )
    test_dataset = AgriculturalDataset(
        data['test']['sequences'],
        data['test']['targets'],
        data['test']['node_indices']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    print("\nInitializing model...")
    model = HybridModel(
        temporal_input_size=8,
        temporal_hidden_size=64,
        node_features=node_features.shape[1],
        graph_hidden_channels=64,
        num_heads=4,
        fusion_hidden_size=128,
        num_inputs=5,
        dropout=0.2
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, config['device'], node_features, edge_index)
        train_history.append(train_metrics)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config['device'], node_features, edge_index, preprocessor)
        val_history.append(val_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['total_loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Individual: {train_metrics['individual_loss']:.4f}, "
              f"Total: {train_metrics['total_cost_loss']:.4f})")
        print(f"Val Loss: {val_metrics['total_loss']:.4f} "
              f"(Individual: {val_metrics['individual_loss']:.4f}, "
              f"Total: {val_metrics['total_cost_loss']:.4f})")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            model_path = os.path.join(config['save_dir'], f"{config['model_name']}_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'config': config
            }, model_path)
            print(f"Saved best model (Val Loss: {best_val_loss:.4f})")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_metrics = validate(model, test_loader, criterion, config['device'], node_features, edge_index, preprocessor)
    
    print("\nTest Metrics:")
    for cost_type, metrics in test_metrics['metrics'].items():
        print(f"\n{cost_type.upper()}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²: {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")
    
    # Save final model and preprocessor
    final_model_path = os.path.join(config['save_dir'], f"{config['model_name']}_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': config['epochs'],
        'test_metrics': test_metrics,
        'config': config
    }, final_model_path)
    
    # Save preprocessor
    import pickle
    preprocessor_path = os.path.join(config['save_dir'], 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save training history
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_history': train_history,
            'val_history': val_history,
            'test_metrics': test_metrics
        }, f, indent=2)
    
    print(f"\nModel saved to: {final_model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Training history saved to: {history_path}")
    print("\nTraining completed!")

if __name__ == '__main__':
    main()

