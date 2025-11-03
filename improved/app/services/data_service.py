import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataService:
    def __init__(self):
        self.crop_district_mapping = {
            'Kampala': ['Matooke', 'Beans', 'Maize', 'Cassava'],
            'Wakiso': ['Matooke', 'Beans', 'Maize', 'Coffee', 'Cassava'],
            'Jinja': ['Coffee', 'Matooke', 'Maize', 'Beans'],
            'Mbale': ['Coffee', 'Matooke', 'Beans', 'Maize'],
            'Gulu': ['Maize', 'Beans', 'Cassava'],
            'Lira': ['Maize', 'Beans', 'Cassava'],
            'Luweero': ['Matooke', 'Maize', 'Beans', 'Cassava', 'Coffee']
        }
        
        self.region_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder() 
        self.crop_encoder = LabelEncoder()
        
        # Initialize with default values
        self.region_encoder.fit(['Central', 'Eastern', 'Northern', 'Western'])
        self.district_encoder.fit(list(self.crop_district_mapping.keys()))
        self.crop_encoder.fit(['Maize', 'Beans', 'Coffee', 'Rice', 'Cassava', 'Matooke'])

    def load_data(self, data_files):
        """Load data from CSV files if available"""
        try:
            # Try to load datasets if they exist
            datasets = []
            for file_type, file_path in data_files.items():
                try:
                    df = pd.read_csv(file_path)
                    datasets.append(df)
                    print(f"Loaded {file_path}")
                except FileNotFoundError:
                    print(f"File not found: {file_path}, using defaults")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            if datasets:
                all_data = pd.concat(datasets, ignore_index=True)
                print(f"Successfully loaded {len(datasets)} dataset files")
            else:
                print("No dataset files found, using default values")
                
        except Exception as e:
            print(f"Error in load_data: {e}")
        
        return True

    def get_districts_for_region(self, region):
        """Get districts for a region"""
        return list(self.crop_district_mapping.keys())

    def get_crops_for_district(self, district):
        """Get crops for a district"""
        return self.crop_district_mapping.get(district, ['Maize', 'Beans', 'Cassava'])