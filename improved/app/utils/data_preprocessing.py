import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch_geometric.data import Data
import torch
from collections import defaultdict

class AgriculturalDataPreprocessor:
    """
    Preprocesses agricultural data for hybrid model training
    Handles temporal sequences and knowledge graph construction
    """
    
    def __init__(self, seq_length=12):  # 12 months lookback
        self.seq_length = seq_length
        self.region_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        
        # Scalers
        self.rainfall_scaler = StandardScaler()
        self.soil_scaler = StandardScaler()
        self.cost_scalers = {
            'seed': StandardScaler(),
            'fertilizer': StandardScaler(),
            'herbicide': StandardScaler(),
            'pesticide': StandardScaler(),
            'labor': StandardScaler()
        }
        
        self.fitted = False
        
    def fit(self, train_df):
        """Fit encoders and scalers on training data"""
        # Fit encoders
        self.region_encoder.fit(train_df['Region'].unique())
        self.district_encoder.fit(train_df['District'].unique())
        self.crop_encoder.fit(train_df['Crop'].unique())
        
        # Fit scalers
        self.rainfall_scaler.fit(train_df[['Rainfall_Index']])
        self.soil_scaler.fit(train_df[['Soil_Fertility_Index']])
        
        # Fit cost scalers
        self.cost_scalers['seed'].fit(train_df[['Seed_Price_Per_Kg']])
        self.cost_scalers['fertilizer'].fit(train_df[['Fertilizer_Price_Per_Kg']])
        self.cost_scalers['herbicide'].fit(train_df[['Herbicide_Price_Per_Litre']])
        self.cost_scalers['pesticide'].fit(train_df[['Pesticide_Price_Per_Litre']])
        self.cost_scalers['labor'].fit(train_df[['Labor_Cost_Per_Day']])
        
        self.fitted = True
        
    def create_knowledge_graph(self, df):
        """
        Create knowledge graph connecting Region -> District -> Crop
        Returns node features and edge indices
        """
        # Create unique nodes for each region, district, and crop combination
        unique_combos = df[['Region', 'District', 'Crop']].drop_duplicates()
        
        # Map each combination to a node index
        node_mapping = {}
        for idx, (_, row) in enumerate(unique_combos.iterrows()):
            combo_key = (row['Region'], row['District'], row['Crop'])
            node_mapping[combo_key] = idx
        
        # Create node features: [region_encoded, district_encoded, crop_encoded]
        num_nodes = len(node_mapping)
        node_features = np.zeros((num_nodes, 3))
        
        for combo_key, node_idx in node_mapping.items():
            region, district, crop = combo_key
            node_features[node_idx, 0] = self.region_encoder.transform([region])[0]
            node_features[node_idx, 1] = self.district_encoder.transform([district])[0]
            node_features[node_idx, 2] = self.crop_encoder.transform([crop])[0]
        
        # Normalize node features
        node_features = node_features / (node_features.max(axis=0) + 1e-8)
        
        # Expand node features with embeddings
        # Convert to one-hot and then use learned embeddings
        region_onehot = np.zeros((num_nodes, len(self.region_encoder.classes_)))
        district_onehot = np.zeros((num_nodes, len(self.district_encoder.classes_)))
        crop_onehot = np.zeros((num_nodes, len(self.crop_encoder.classes_)))
        
        for combo_key, node_idx in node_mapping.items():
            region, district, crop = combo_key
            region_idx = self.region_encoder.transform([region])[0]
            district_idx = self.district_encoder.transform([district])[0]
            crop_idx = self.crop_encoder.transform([crop])[0]
            
            region_onehot[node_idx, region_idx] = 1.0
            district_onehot[node_idx, district_idx] = 1.0
            crop_onehot[node_idx, crop_idx] = 1.0
        
        # Concatenate features: 3 base + one-hot encodings
        expanded_features = np.hstack([
            node_features,
            region_onehot,
            district_onehot,
            crop_onehot
        ])
        
        # Create edges: connect regions to districts, districts to crops
        edges = []
        
        # Add edges between regions and districts
        region_to_district = defaultdict(set)
        for combo_key in node_mapping.keys():
            region, district, _ = combo_key
            region_to_district[region].add(district)
        
        # Add edges between districts and crops
        district_to_crop = defaultdict(set)
        for combo_key in node_mapping.keys():
            region, district, crop = combo_key
            district_to_crop[(region, district)].add(crop)
        
        # Create graph edges: connect nodes that share regions/districts
        # Strategy: connect all nodes in the same region, and nodes with same district
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                combo_i = unique_combos.iloc[i]
                combo_j = unique_combos.iloc[j]
                
                # Connect if same region
                if combo_i['Region'] == combo_j['Region']:
                    edges.append([i, j])
                    edges.append([j, i])
                # Connect if same district
                elif combo_i['District'] == combo_j['District']:
                    edges.append([i, j])
                    edges.append([j, i])
                # Connect if same crop type
                elif combo_i['Crop'] == combo_j['Crop']:
                    edges.append([i, j])
                    edges.append([j, i])
        
        edge_index = np.array(edges).T if edges else np.array([[], []])
        
        return {
            'node_features': torch.FloatTensor(expanded_features),
            'edge_index': torch.LongTensor(edge_index),
            'node_mapping': node_mapping,
            'unique_combos': unique_combos
        }
    
    def create_temporal_sequences(self, df, graph_info):
        """
        Create temporal sequences for each unique (Region, District, Crop) combination
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Region', 'District', 'Crop', 'Date'])
        
        sequences = []
        targets = []
        node_indices = []
        
        unique_combos = graph_info['unique_combos']
        node_mapping = graph_info['node_mapping']
        
        for idx, combo in unique_combos.iterrows():
            region, district, crop = combo['Region'], combo['District'], combo['Crop']
            combo_key = (region, district, crop)
            node_idx = node_mapping[combo_key]
            
            # Filter data for this combination
            combo_data = df[
                (df['Region'] == region) & 
                (df['District'] == district) & 
                (df['Crop'] == crop)
            ].reset_index(drop=True)
            
            if len(combo_data) < self.seq_length + 1:
                continue  # Skip if not enough data
            
            # Create sequences
            for i in range(len(combo_data) - self.seq_length):
                seq_data = combo_data.iloc[i:i + self.seq_length]
                target_data = combo_data.iloc[i + self.seq_length]
                
                # Temporal features: [Rainfall, Soil, Month, Year, region_enc, district_enc, crop_enc]
                temporal_seq = []
                for _, row in seq_data.iterrows():
                    # Use DataFrame format to avoid sklearn warnings
                    rainfall_val = self.rainfall_scaler.transform(pd.DataFrame([[row['Rainfall_Index']]], columns=['Rainfall_Index']))[0, 0]
                    soil_val = self.soil_scaler.transform(pd.DataFrame([[row['Soil_Fertility_Index']]], columns=['Soil_Fertility_Index']))[0, 0]
                    
                    features = [
                        rainfall_val,
                        soil_val,
                        row['Date'].month / 12.0,  # Normalized month
                        (row['Date'].year - 2018) / 10.0,  # Normalized year
                        self.region_encoder.transform([region])[0] / len(self.region_encoder.classes_),
                        self.district_encoder.transform([district])[0] / len(self.district_encoder.classes_),
                        self.crop_encoder.transform([crop])[0] / len(self.crop_encoder.classes_),
                        0.0  # Placeholder for future expansion
                    ]
                    temporal_seq.append(features)
                
                # Targets (actual costs) - use DataFrame format
                target = {
                    'seed': self.cost_scalers['seed'].transform(pd.DataFrame([[target_data['Seed_Price_Per_Kg']]], columns=['Seed_Price_Per_Kg']))[0, 0],
                    'fertilizer': self.cost_scalers['fertilizer'].transform(pd.DataFrame([[target_data['Fertilizer_Price_Per_Kg']]], columns=['Fertilizer_Price_Per_Kg']))[0, 0],
                    'herbicide': self.cost_scalers['herbicide'].transform(pd.DataFrame([[target_data['Herbicide_Price_Per_Litre']]], columns=['Herbicide_Price_Per_Litre']))[0, 0],
                    'pesticide': self.cost_scalers['pesticide'].transform(pd.DataFrame([[target_data['Pesticide_Price_Per_Litre']]], columns=['Pesticide_Price_Per_Litre']))[0, 0],
                    'labor': self.cost_scalers['labor'].transform(pd.DataFrame([[target_data['Labor_Cost_Per_Day']]], columns=['Labor_Cost_Per_Day']))[0, 0]
                }
                
                # Calculate total cost if available, otherwise sum individual costs
                # Note: Total cost is the sum of individual costs (not scaled)
                raw_total = (
                    target_data['Seed_Price_Per_Kg'] +
                    target_data['Fertilizer_Price_Per_Kg'] +
                    target_data['Herbicide_Price_Per_Litre'] +
                    target_data['Pesticide_Price_Per_Litre'] +
                    target_data['Labor_Cost_Per_Day']
                )
                
                # Store both raw and for comparison with validation/test
                if 'Total_Input_Cost' in target_data:
                    target['total'] = target_data['Total_Input_Cost']  # Use provided total if available
                    target['total_raw'] = raw_total  # Also store raw sum for reference
                else:
                    target['total'] = raw_total  # Use raw sum for training
                
                sequences.append(np.array(temporal_seq))
                targets.append(target)
                node_indices.append(node_idx)
        
        return {
            'sequences': torch.FloatTensor(np.array(sequences)),
            'targets': targets,
            'node_indices': torch.LongTensor(node_indices)
        }
    
    def prepare_data(self, train_df, val_df=None, test_df=None):
        """
        Prepare all datasets for training
        """
        if not self.fitted:
            self.fit(train_df)
        
        # Create knowledge graph
        graph_info = self.create_knowledge_graph(train_df)
        
        # Create temporal sequences
        train_data = self.create_temporal_sequences(train_df, graph_info)
        val_data = self.create_temporal_sequences(val_df, graph_info) if val_df is not None else None
        test_data = self.create_temporal_sequences(test_df, graph_info) if test_df is not None else None
        
        return {
            'graph': graph_info,
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def inverse_transform_costs(self, predictions):
        """Convert normalized predictions back to original scale"""
        result = {}
        for key, scaler_key in [('seed', 'seed'), ('fertilizer', 'fertilizer'), 
                                ('herbicide', 'herbicide'), ('pesticide', 'pesticide'), 
                                ('labor', 'labor')]:
            p = predictions[key]
            if isinstance(p, (int, float, np.number)):
                # Single value
                val = pd.DataFrame([[p]], columns=[f'{key.capitalize()}_Price'])
                # Use appropriate column name based on scaler
                col_names = {
                    'seed': 'Seed_Price_Per_Kg',
                    'fertilizer': 'Fertilizer_Price_Per_Kg',
                    'herbicide': 'Herbicide_Price_Per_Litre',
                    'pesticide': 'Pesticide_Price_Per_Litre',
                    'labor': 'Labor_Cost_Per_Day'
                }
                val = pd.DataFrame([[p]], columns=[col_names[scaler_key]])
                result[key] = self.cost_scalers[scaler_key].inverse_transform(val)[0, 0]
            else:
                # Array of values
                col_names = {
                    'seed': 'Seed_Price_Per_Kg',
                    'fertilizer': 'Fertilizer_Price_Per_Kg',
                    'herbicide': 'Herbicide_Price_Per_Litre',
                    'pesticide': 'Pesticide_Price_Per_Litre',
                    'labor': 'Labor_Cost_Per_Day'
                }
                val = pd.DataFrame(p.reshape(-1, 1), columns=[col_names[scaler_key]])
                result[key] = self.cost_scalers[scaler_key].inverse_transform(val).flatten()
        return result

