"""
Data cleaning and preprocessing module for car dataset.
Handles missing values, data validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataCleaner:
    """Cleans and validates car dataset."""
    
    def __init__(self, csv_path):
        """Initialize with CSV file path."""
        self.csv_path = csv_path
        self.df = None
    
    def load_data(self):
        """Load CSV file with error handling."""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded {len(self.df)} cars from dataset")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean(self):
        """
        Clean the dataset:
        - Handle missing values
        - Remove duplicates
        - Validate data types
        - Standardize text fields
        """
        if self.df is None:
            self.load_data()
        
        original_count = len(self.df)
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Fill numeric missing values with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill categorical missing values with 'Unknown'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col].fillna('Unknown', inplace=True)
        
        # Remove duplicate cars
        self.df = self.df.drop_duplicates(subset=['Brand', 'Model', 'Year'], keep='first')
        
        # Standardize text fields
        self.df['Brand'] = self.df['Brand'].str.strip().str.title()
        self.df['Model'] = self.df['Model'].str.strip().str.title()
        self.df['Fuel_Type'] = self.df['Fuel_Type'].str.strip().str.capitalize()
        self.df['Transmission'] = self.df['Transmission'].str.strip().str.capitalize()
        
        # Ensure numeric columns are correct type
        numeric_cols_to_convert = ['Price', 'Mileage', 'Engine_CC', 
                                    'Seating_Capacity', 'Service_Cost']
        for col in numeric_cols_to_convert:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove rows with invalid prices or zero prices
        if 'Price' in self.df.columns:
            self.df = self.df[self.df['Price'] > 0]
        
        # Add calculated features
        self._add_features()
        
        print(f"✓ Cleaned data: {original_count} → {len(self.df)} cars")
        print(f"  - Removed {original_count - len(self.df)} invalid/duplicate rows")
        
        return self.df
    
    def _add_features(self):
        """Add useful calculated features for recommendations."""
        
        # Price category
        self.df['Price_Category'] = pd.cut(
            self.df['Price'],
            bins=[0, 500000, 1000000, 1500000, 2000000, float('inf')],
            labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury']
        )
        
        # Vehicle age
        from datetime import datetime
        current_year = datetime.now().year
        self.df['Age'] = current_year - self.df['Year']
        
        # Fuel type categories for easy filtering
        self.df['Fuel_Type'] = self.df['Fuel_Type'].replace({
            'Petrol': 'Petrol',
            'Diesel': 'Diesel',
            'Cng': 'CNG',
            'Electric': 'Electric',
            'Lpg': 'LPG'
        })
    
    def get_clean_data(self):
        """Return cleaned dataframe."""
        if self.df is None:
            self.load_data()
            self.clean()
        return self.df
    
    def save_cleaned_data(self, output_path='cleaned_cars.csv'):
        """Save cleaned data to CSV."""
        if self.df is None:
            raise ValueError("No data to save. Run clean() first.")
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved cleaned data to {output_path}")


def prepare_dataset(csv_path, output_path='cleaned_cars.csv'):
    """
    Convenience function to prepare dataset in one call.
    
    Usage:
        df = prepare_dataset('car_dataset_india.csv')
    """
    cleaner = DataCleaner(r'D:\\car_chatbot_2\\data\\car_dataset_india.csv')
    cleaner.load_data()
    cleaner.clean()
    cleaner.save_cleaned_data(r'D:\\car_chatbot_2\\data\\cleaned_cars.csv')
    return cleaner.get_clean_data()

