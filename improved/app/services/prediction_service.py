"""
Prediction Service using the Hybrid Model (Temporal NN + Knowledge Graph)
Predicts individual agricultural input costs and aggregates to total cost
"""
import os
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder

class PredictionService:
    def __init__(self, model=None, preprocessor=None, graph_info=None):
        """
        Initialize prediction service with trained model
        
        Args:
            model: Trained HybridModel instance
            preprocessor: AgriculturalDataPreprocessor instance
            graph_info: Graph information (node_features, edge_index, node_mapping)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.graph_info = graph_info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)
    
    def load_model_from_path(self, model_path, preprocessor_path, graph_data_path=None):
        """
        Load model and preprocessor from file paths
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Path to saved preprocessor
            graph_data_path: Optional path to saved graph data
        """
        try:
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Load graph info if saved separately, otherwise reconstruct
            if graph_data_path and os.path.exists(graph_data_path):
                with open(graph_data_path, 'rb') as f:
                    self.graph_info = pickle.load(f)
            else:
                # Reconstruct graph from training data
                train_df = pd.read_csv('train_dataset_cleaned.csv')
                self.graph_info = self.preprocessor.create_knowledge_graph(train_df)
            
            # Load model
            from app.models.hybrid_model import HybridModel
            
            node_features_dim = self.graph_info['node_features'].shape[1]
            
            self.model = HybridModel(
                temporal_input_size=8,
                temporal_hidden_size=64,
                node_features=node_features_dim,
                graph_hidden_channels=64,
                num_heads=4,
                fusion_hidden_size=128,
                num_inputs=5,
                dropout=0.2
            ).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_input_sequence(self, region, district, crop, historical_data=None, seq_length=12):
        """
        Prepare temporal input sequence for prediction
        
        Args:
            region: Region name
            district: District name
            crop: Crop name
            historical_data: Optional DataFrame with historical data
            seq_length: Length of temporal sequence
        """
        if historical_data is None:
            # Load historical data from training set
            train_df = pd.read_csv('train_dataset_cleaned.csv')
            historical_data = train_df[
                (train_df['Region'] == region) &
                (train_df['District'] == district) &
                (train_df['Crop'] == crop)
            ].sort_values('Date').tail(seq_length)
        
        if len(historical_data) < seq_length:
            # Pad with last available data
            last_row = historical_data.iloc[-1] if len(historical_data) > 0 else None
            if last_row is None:
                raise ValueError(f"No historical data found for {region}, {district}, {crop}")
            
            # Replicate last row to fill sequence
            historical_data = pd.concat([historical_data] + [pd.DataFrame([last_row])] * (seq_length - len(historical_data)))
            historical_data = historical_data.tail(seq_length)
        
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        # Create temporal sequence
        temporal_seq = []
        for _, row in historical_data.tail(seq_length).iterrows():
            # Use DataFrame format to match training preprocessing
            rainfall_val = self.preprocessor.rainfall_scaler.transform(
                pd.DataFrame([[row['Rainfall_Index']]], columns=['Rainfall_Index'])
            )[0, 0]
            soil_val = self.preprocessor.soil_scaler.transform(
                pd.DataFrame([[row['Soil_Fertility_Index']]], columns=['Soil_Fertility_Index'])
            )[0, 0]
            
            features = [
                rainfall_val,
                soil_val,
                row['Date'].month / 12.0,
                (row['Date'].year - 2018) / 10.0,
                self.preprocessor.region_encoder.transform([region])[0] / len(self.preprocessor.region_encoder.classes_),
                self.preprocessor.district_encoder.transform([district])[0] / len(self.preprocessor.district_encoder.classes_),
                self.preprocessor.crop_encoder.transform([crop])[0] / len(self.preprocessor.crop_encoder.classes_),
                0.0
            ]
            temporal_seq.append(features)
        
        return np.array(temporal_seq)
    
    def get_node_index(self, region, district, crop):
        """Get graph node index for a region-district-crop combination"""
        combo_key = (region, district, crop)
        node_mapping = self.graph_info['node_mapping']
        
        if combo_key in node_mapping:
            return node_mapping[combo_key]
        else:
            # If not found, use a default node (first node)
            print(f"Warning: Node not found for {combo_key}, using default node")
            return 0
    
    def predict_for_future_month(self, region, district, crop, months_ahead=1, historical_data=None):
        """
        Predict costs for a specific month ahead by adjusting temporal sequence
        
        Args:
            region: Region name
            district: District name
            crop: Crop name
            months_ahead: Number of months to predict ahead (1 = next month)
            historical_data: Optional historical data
            
        Returns:
            dict: Predictions for that month
        """
        if self.model is None or self.preprocessor is None or self.graph_info is None:
            raise ValueError("Model, preprocessor, or graph info not loaded. Call load_model_from_path() first.")
        
        # Get base historical data
        if historical_data is None:
            train_df = pd.read_csv('train_dataset_cleaned.csv')
            historical_data = train_df[
                (train_df['Region'] == region) &
                (train_df['District'] == district) &
                (train_df['Crop'] == crop)
            ].sort_values('Date').tail(12)
        
        if len(historical_data) == 0:
            raise ValueError(f"No historical data found for {region}, {district}, {crop}")
        
        historical_data = historical_data.copy()
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        # Create a shifted sequence for the future month
        # Properly shift the entire sequence forward by months_ahead
        last_historical_date = historical_data['Date'].iloc[-1]
        future_date = last_historical_date + relativedelta(months=months_ahead)
        
        # Shift the entire sequence forward properly
        # Each prediction should see the sequence shifted forward by months_ahead
        # This creates different temporal contexts for the model
        
        # Get the base sequence (last 12 months of historical data)
        base_sequence = historical_data.tail(12).copy()
        
        if len(base_sequence) < 12:
            # If we don't have enough data, pad by repeating the earliest row
            needed = 12 - len(base_sequence)
            earliest_row = base_sequence.iloc[0]
            padding_rows = []
            for i in range(needed, 0, -1):
                pad_row = earliest_row.copy()
                pad_row['Date'] = earliest_row['Date'] - relativedelta(months=i)
                padding_rows.append(pd.DataFrame([pad_row]))
            base_sequence = pd.concat(padding_rows + [base_sequence]).reset_index(drop=True)
        
        # Shift the entire sequence forward by months_ahead
        # This creates a temporal sequence where the model sees the "future" context
        seq_data = base_sequence.copy()
        date_col_idx = seq_data.columns.get_loc('Date')
        
        for idx in range(len(seq_data)):
            # Shift each date forward by months_ahead
            old_date = seq_data.iloc[idx]['Date']
            new_date = old_date + relativedelta(months=months_ahead)
            seq_data.iloc[idx, date_col_idx] = new_date
        
        # The last entry should be exactly the future_date we want to predict
        seq_data.iloc[-1, date_col_idx] = future_date
        
        # Sort to ensure chronological order
        seq_data = seq_data.sort_values('Date').reset_index(drop=True)
        
        # Prepare temporal sequence with future date
        temporal_seq = []
        for _, row in seq_data.iterrows():
            rainfall_val = self.preprocessor.rainfall_scaler.transform(
                pd.DataFrame([[row['Rainfall_Index']]], columns=['Rainfall_Index'])
            )[0, 0]
            soil_val = self.preprocessor.soil_scaler.transform(
                pd.DataFrame([[row['Soil_Fertility_Index']]], columns=['Soil_Fertility_Index'])
            )[0, 0]
            
            features = [
                rainfall_val,
                soil_val,
                row['Date'].month / 12.0,  # This will reflect the future month
                (row['Date'].year - 2018) / 10.0,  # This will reflect the future year
                self.preprocessor.region_encoder.transform([region])[0] / len(self.preprocessor.region_encoder.classes_),
                self.preprocessor.district_encoder.transform([district])[0] / len(self.preprocessor.district_encoder.classes_),
                self.preprocessor.crop_encoder.transform([crop])[0] / len(self.preprocessor.crop_encoder.classes_),
                0.0
            ]
            temporal_seq.append(features)
        
        # Get node index
        node_idx = self.get_node_index(region, district, crop)
        
        # Prepare tensors
        temporal_features = torch.FloatTensor(np.array(temporal_seq)).unsqueeze(0).to(self.device)
        node_features = self.graph_info['node_features'].to(self.device)
        edge_index = self.graph_info['edge_index'].to(self.device)
        node_indices = torch.LongTensor([node_idx]).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(temporal_features, node_features, edge_index, node_indices)
        
        # Inverse transform to original scale
        denormalized = self.preprocessor.inverse_transform_costs({
            'seed': predictions['seed'].cpu().numpy()[0],
            'fertilizer': predictions['fertilizer'].cpu().numpy()[0],
            'herbicide': predictions['herbicide'].cpu().numpy()[0],
            'pesticide': predictions['pesticide'].cpu().numpy()[0],
            'labor': predictions['labor'].cpu().numpy()[0]
        })
        
        # Calculate total cost
        total_cost = (
            denormalized['seed'] +
            denormalized['fertilizer'] +
            denormalized['herbicide'] +
            denormalized['pesticide'] +
            denormalized['labor']
        )
        
        return {
            'seed_price_per_kg': float(denormalized['seed']),
            'fertilizer_price_per_kg': float(denormalized['fertilizer']),
            'herbicide_price_per_litre': float(denormalized['herbicide']),
            'pesticide_price_per_litre': float(denormalized['pesticide']),
            'labor_cost_per_day': float(denormalized['labor']),
            'total_input_cost': float(total_cost),
            'region': region,
            'district': district,
            'crop': crop
        }
    
    def predict_individual_costs(self, region, district, crop, historical_data=None):
        """
        Predict individual agricultural input costs using the hybrid model
        
        Args:
            region: Region name
            district: District name
            crop: Crop name
            historical_data: Optional historical data for temporal features
            
        Returns:
            dict: Predictions for each input cost (seed, fertilizer, herbicide, pesticide, labor) and total
        """
        if self.model is None or self.preprocessor is None or self.graph_info is None:
            raise ValueError("Model, preprocessor, or graph info not loaded. Call load_model_from_path() first.")
        
        # Prepare input sequence
        temporal_seq = self.prepare_input_sequence(region, district, crop, historical_data)
        
        # Get node index
        node_idx = self.get_node_index(region, district, crop)
        
        # Prepare tensors
        temporal_features = torch.FloatTensor(temporal_seq).unsqueeze(0).to(self.device)  # (1, seq_length, 8)
        node_features = self.graph_info['node_features'].to(self.device)
        edge_index = self.graph_info['edge_index'].to(self.device)
        node_indices = torch.LongTensor([node_idx]).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(temporal_features, node_features, edge_index, node_indices)
        
        # Inverse transform to original scale
        denormalized = self.preprocessor.inverse_transform_costs({
            'seed': predictions['seed'].cpu().numpy()[0],
            'fertilizer': predictions['fertilizer'].cpu().numpy()[0],
            'herbicide': predictions['herbicide'].cpu().numpy()[0],
            'pesticide': predictions['pesticide'].cpu().numpy()[0],
            'labor': predictions['labor'].cpu().numpy()[0]
        })
        
        # Calculate total cost
        total_cost = (
            denormalized['seed'] +
            denormalized['fertilizer'] +
            denormalized['herbicide'] +
            denormalized['pesticide'] +
            denormalized['labor']
        )
        
        return {
            'seed_price_per_kg': float(denormalized['seed']),
            'fertilizer_price_per_kg': float(denormalized['fertilizer']),
            'herbicide_price_per_litre': float(denormalized['herbicide']),
            'pesticide_price_per_litre': float(denormalized['pesticide']),
            'labor_cost_per_day': float(denormalized['labor']),
            'total_input_cost': float(total_cost),
            'region': region,
            'district': district,
            'crop': crop
        }
    
    def predict_future_costs(self, region, district, crop, months_ahead=1, historical_data=None):
        """
        Predict future costs for multiple months ahead
        
        Args:
            region: Region name
            district: District name
            crop: Crop name
            months_ahead: Number of months to predict
            historical_data: Optional historical data
            
        Returns:
            list: Predictions for each month
        """
        predictions = []
        
        # Get base prediction
        base_prediction = self.predict_individual_costs(region, district, crop, historical_data)
        
        for month_offset in range(1, months_ahead + 1):
            future_date = datetime.now() + timedelta(days=30 * month_offset)
            
            # For simplicity, use base prediction with slight adjustments
            # In production, you might want to use the predicted costs as historical data
            # and recursively predict future months
            
            prediction = {
                'month': future_date.strftime("%B %Y"),
                'date': future_date.strftime("%Y-%m-%d"),
                'seed_price_per_kg': base_prediction['seed_price_per_kg'],
                'fertilizer_price_per_kg': base_prediction['fertilizer_price_per_kg'],
                'herbicide_price_per_litre': base_prediction['herbicide_price_per_litre'],
                'pesticide_price_per_litre': base_prediction['pesticide_price_per_litre'],
                'labor_cost_per_day': base_prediction['labor_cost_per_day'],
                'total_input_cost': base_prediction['total_input_cost'],
                'region': region,
                'district': district,
                'crop': crop
            }
            
            predictions.append(prediction)
        
        return predictions
