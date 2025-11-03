import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np

class TemporalNN(nn.Module):
    """
    Temporal Neural Network for time-series patterns
    Uses LSTM to capture temporal dependencies in agricultural input costs
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(TemporalNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size) - temporal features
        Returns:
            output: (batch_size, hidden_size) - temporal representation
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last output from the sequence
        return lstm_out[:, -1, :]  # (batch_size, hidden_size)


class KnowledgeGraphNN(nn.Module):
    """
    Knowledge Graph Neural Network using Graph Attention Networks (GAT)
    Captures relationships between Region, District, and Crop entities
    """
    def __init__(self, node_features, hidden_channels, num_heads=4, dropout=0.2):
        super(KnowledgeGraphNN, self).__init__()
        
        self.gat1 = GATConv(
            in_channels=node_features,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: (num_nodes, node_features) - node features
            edge_index: (2, num_edges) - graph edge connections
        Returns:
            output: (num_nodes, hidden_channels) - graph representation
        """
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        return x


class HybridModel(nn.Module):
    """
    Hybrid Model combining Temporal NN and Knowledge Graph
    Predicts individual agricultural input costs and aggregates to total cost
    """
    def __init__(
        self,
        temporal_input_size=8,  # temporal features (weather, soil, etc.)
        temporal_hidden_size=64,
        node_features=32,  # graph node features (encoded region, district, crop)
        graph_hidden_channels=64,
        num_heads=4,
        fusion_hidden_size=128,
        num_inputs=5,  # seed, fertilizer, herbicide, pesticide, labor
        dropout=0.2
    ):
        super(HybridModel, self).__init__()
        
        # Temporal component
        self.temporal_nn = TemporalNN(
            input_size=temporal_input_size,
            hidden_size=temporal_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        # Knowledge Graph component
        self.graph_nn = KnowledgeGraphNN(
            node_features=node_features,
            hidden_channels=graph_hidden_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion layer to combine temporal and graph features
        fusion_input_size = temporal_hidden_size + graph_hidden_channels
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads for each input cost
        self.seed_head = nn.Linear(fusion_hidden_size // 2, 1)
        self.fertilizer_head = nn.Linear(fusion_hidden_size // 2, 1)
        self.herbicide_head = nn.Linear(fusion_hidden_size // 2, 1)
        self.pesticide_head = nn.Linear(fusion_hidden_size // 2, 1)
        self.labor_head = nn.Linear(fusion_hidden_size // 2, 1)
        
    def forward(self, temporal_features, node_features, edge_index, node_indices):
        """
        Args:
            temporal_features: (batch_size, seq_length, temporal_input_size) - time-series features
            node_features: (num_nodes, node_features) - graph node features
            edge_index: (2, num_edges) - graph edges
            node_indices: (batch_size,) - indices mapping batch samples to graph nodes
        Returns:
            predictions: dict with individual cost predictions and total cost
        """
        # Temporal representation
        temporal_repr = self.temporal_nn(temporal_features)  # (batch_size, temporal_hidden_size)
        
        # Graph representation for all nodes
        graph_repr = self.graph_nn(node_features, edge_index)  # (num_nodes, graph_hidden_channels)
        
        # Select graph representation for batch samples
        batch_graph_repr = graph_repr[node_indices]  # (batch_size, graph_hidden_channels)
        
        # Fusion of temporal and graph features
        fused_features = torch.cat([temporal_repr, batch_graph_repr], dim=1)  # (batch_size, fusion_input_size)
        fused_repr = self.fusion(fused_features)  # (batch_size, fusion_hidden_size // 2)
        
        # Individual cost predictions
        seed_pred = F.relu(self.seed_head(fused_repr)).squeeze(-1)
        fertilizer_pred = F.relu(self.fertilizer_head(fused_repr)).squeeze(-1)
        herbicide_pred = F.relu(self.herbicide_head(fused_repr)).squeeze(-1)
        pesticide_pred = F.relu(self.pesticide_head(fused_repr)).squeeze(-1)
        labor_pred = F.relu(self.labor_head(fused_repr)).squeeze(-1)
        
        # Aggregate to total cost
        total_cost = seed_pred + fertilizer_pred + herbicide_pred + pesticide_pred + labor_pred
        
        return {
            'seed': seed_pred,
            'fertilizer': fertilizer_pred,
            'herbicide': herbicide_pred,
            'pesticide': pesticide_pred,
            'labor': labor_pred,
            'total': total_cost
        }


