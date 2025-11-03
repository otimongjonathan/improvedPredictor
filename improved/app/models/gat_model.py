"""
Legacy GAT model - kept for backward compatibility
The new hybrid model is in hybrid_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os

# Import the new hybrid model
from .hybrid_model import HybridModel

class HybridGATModel(nn.Module):
    """Legacy model - use HybridModel instead"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HybridGATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=True)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze()

# Global model instance
models = {}

def init_models(app):
    """Initialize hybrid model for the application"""
    try:
        device = torch.device(app.config['DEVICE'])
        
        # Try to load preprocessor to get model dimensions
        import pickle
        preprocessor_path = app.config.get('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
        
        node_features_dim = 32  # Default
        if os.path.exists(preprocessor_path):
            try:
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                    # Calculate node features dimension
                    region_classes = len(preprocessor.region_encoder.classes_)
                    district_classes = len(preprocessor.district_encoder.classes_)
                    crop_classes = len(preprocessor.crop_encoder.classes_)
                    node_features_dim = 3 + region_classes + district_classes + crop_classes
            except Exception as e:
                app.logger.warning(f"Could not load preprocessor: {e}")
        
        # Initialize hybrid model architecture
        models['hybrid'] = HybridModel(
            temporal_input_size=8,
            temporal_hidden_size=64,
            node_features=node_features_dim,
            graph_hidden_channels=64,
            num_heads=4,
            fusion_hidden_size=128,
            num_inputs=5,
            dropout=0.2
        ).to(device)
        
        # Load trained weights if available
        model_file = app.config.get('MODEL_PATH', 'models/hybrid_agricultural_model_best.pth')
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(model_file, map_location=device)
                if 'model_state_dict' in checkpoint:
                    models['hybrid'].load_state_dict(checkpoint['model_state_dict'])
                else:
                    models['hybrid'].load_state_dict(checkpoint)
                models['hybrid'].eval()
                app.logger.info(f"Hybrid model loaded successfully from {model_file}")
            except Exception as e:
                app.logger.warning(f"Error loading model weights: {e}")
                app.logger.info("Using newly initialized model")
        else:
            app.logger.info(f"Model file not found at {model_file}. Using newly initialized model")
            
    except Exception as e:
        app.logger.error(f"Error initializing model: {e}")