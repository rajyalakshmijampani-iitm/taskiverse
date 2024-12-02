# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "seaborn",
#     "matplotlib",
#     "scipy",
#     "scikit-learn"
# ]
# ///

import os
import sys
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def read_csv(file_path):
    """Read the CSV file and return the DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def clean_data(df):
    """Basic data cleaning: delete duplicate rows and empty rows"""
    # Drop duplicate rows
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    return df

def find_unique_id(df):
    """Finding the unique identifier column(s)"""
    
    # Step 1: Check if any single column is unique
    for col in df.columns:
        if df[col].is_unique:
            return [col]
    
    # Step 2: If no single column is unique, check combinations of columns
    for r in range(2, len(df.columns) + 1):  # Start from 2 columns to combination of all
        for combo in combinations(df.columns, r):
            if df[list(combo)].duplicated().sum() == 0:
                return list(combo)
    
    # If no combination is unique, return None
    return None

def identify_column_issues(df):
    """Identify known issues in each column of the dataframe"""
    issues = {}
    
    for col in df.columns:
        column_issues = []
        
        # Check for missing values
        if df[col].isnull().sum() > 0:
            column_issues.append("Missing values")
        
        # Check for duplicates
        if df.duplicated(subset=[col]).sum() > 0:
            column_issues.append("Duplicates")
        
        # Check for outliers using IQR method
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                column_issues.append("Outliers")
        
        # Check for constant values (columns with the same value in all rows)
        if df[col].nunique() == 1:
            column_issues.append("Constant values")
        
        # If no issues were detected, add "No issues"
        if len(column_issues) == 0:
            column_issues.append("No issues")
        
        issues[col] = column_issues
    
    return issues

# Function to generate data context
def generate_data_context(file_name, df):
    """Generates the data context: filename, column names & types, and additional context"""
    context = {}

    # Filename
    context['filename'] = file_name+'.csv'
    
    # Column names and types (convert pandas dtypes to strings)
    context['columns'] = {col: str(df[col].dtype) for col in df.columns}
    
    # Generate summary statistics
    context['summary'] = {col: df[col].describe().to_dict() for col in df.columns}

    # Unique identifier
    context['unique identifier'] = find_unique_id(df)

    # Data issues
    context['known issues'] = identify_column_issues(df)

    return context

# Function to visualize correlation matrix
def plot_correlation_matrix(df):
    """Function to visualize correlation matrix (only numeric columns)."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.shape[1] > 1:  # Ensure there's more than 1 numeric column
        correlation = numeric_df.corr()  # Calculate correlation matrix

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()
    else:
        print("Not enough numeric columns for correlation matrix.")


# Function to detect and visualize outliers using z-score
def detect_outliers(df):
    z_scores = stats.zscore(df.select_dtypes(include='number'))  # Calculate z-scores for numeric columns
    abs_z_scores = abs(z_scores)
    outliers = (abs_z_scores > 3).all(axis=1)  # Identify rows with outliers (z-score > 3)
    outlier_indices = df[outliers].index
    print("\nOutliers detected at indices:", outlier_indices.tolist())

    # Visualize the outliers
    for column in df.select_dtypes(include='number'):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f"Outliers in {column}")
        plt.show()

# Function for clustering (KMeans)
def perform_clustering(df):
    # Standardizing the data before clustering
    df_numeric = df.select_dtypes(include='number').dropna()  # Use numeric columns and drop rows with NaN values
    if len(df_numeric) < 2:
        print("Insufficient data for clustering.")
        return
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Perform KMeans clustering with 3 clusters (adjustable)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_numeric['Cluster'] = kmeans.fit_predict(df_scaled)
    
    print("\nClustering Result:")
    print(df_numeric[['Cluster']].head())  # Show the first few rows with cluster assignments

    # Visualize clustering result if there are only 2 or 3 numeric features
    if df_numeric.shape[1] == 2:
        plt.scatter(df_numeric.iloc[:, 0], df_numeric.iloc[:, 1], c=df_numeric['Cluster'], cmap='viridis')
        plt.title("Clustering Result")
        plt.xlabel(df_numeric.columns[0])
        plt.ylabel(df_numeric.columns[1])
        plt.show()

def create_output_folder(folder_name):
    """Create an output folder for the given dataset name."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Main function to perform all analyses
def perform_generic_analysis(df):

    # Visualize correlation matrix
    plot_correlation_matrix(df)

    # Detect outliers
    detect_outliers(df)

    # Perform clustering
    perform_clustering(df)

def main():
    """Main function to process the input CSV and generate visualizations."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <filename.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print("File not found.")
        sys.exit(1)

    # Extract file name and create a folder with that name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = create_output_folder(file_name)

    # Read and clean the dataset
    df = read_csv(file_path)
    df = clean_data(df)
    print(f"Dataset loaded and cleaned successfully.")
    
    # Generate data context
    context = generate_data_context(file_name, df) 

    # Save data context
    with open(f'{output_folder}/data_context.json', 'w') as context_file:
        json.dump(context, context_file, indent=4)

    print(f"Data context saved to '{output_folder}/data_context.json'")

if __name__ == "__main__":
    main()
