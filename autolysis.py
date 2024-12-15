# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "requests",
#     "Pillow",
#     "tk"
# ]
# ///

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from PIL import Image
import warnings

# Suppress warnings globally
warnings.filterwarnings("ignore")
matplotlib.use('Agg')

# Encapsulated configuration variables
class Config:
    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
    if AIPROXY_TOKEN is None:
        print("Please set token to proceed")
        sys.exit(1)

    URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    MODEL = "gpt-4o-mini"

def summarize_data(df):
    # Generate a summary of the dataset, including basic statistics, advanced metrics, and exploratory analysis insights.
    summary={}
    
    # Basic data cleaning
    df = df.drop_duplicates()
    df = df.dropna(how='all')  # Remove rows with all null values

    # Numeric data summary
    numeric_cols = df.select_dtypes(exclude='object')
    if not numeric_cols.empty:
        numeric_summary = numeric_cols.describe().transpose()
        numeric_summary['missing%'] = numeric_cols.isnull().mean() * 100
        numeric_summary['unique_values'] = numeric_cols.nunique()
        numeric_summary['skewness'] = numeric_cols.skew()
        numeric_summary['kurtosis'] = numeric_cols.kurtosis()
        
        # Detect potential outliers using the IQR rule
        q1 = numeric_cols.quantile(0.25)
        q3 = numeric_cols.quantile(0.75)
        iqr = q3 - q1
        outlier_flags = ((numeric_cols < (q1 - 1.5 * iqr)) | (numeric_cols > (q3 + 1.5 * iqr))).sum()
        numeric_summary['outliers'] = outlier_flags

        summary['numeric'] = numeric_summary

    # Categorical data summary
    categorical_cols = df.select_dtypes(include='object')
    if not categorical_cols.empty:
        categorical_summary = categorical_cols.describe().transpose()
        categorical_summary['missing%'] = categorical_cols.isnull().mean() * 100
        categorical_summary['unique_values'] = categorical_cols.nunique()

        # Detect categorical imbalance
        imbalance_flags = categorical_cols.apply(lambda col: col.value_counts(normalize=True).iloc[0] > 0.8)
        categorical_summary['imbalance'] = imbalance_flags

        summary['categorical'] = categorical_summary

    # Correlation matrix for numeric columns
    if numeric_cols.shape[1] > 1:
        correlation_matrix = numeric_cols.corr()
        high_corr_cols = correlation_matrix.columns[correlation_matrix.abs().gt(0.9).any()]
        correlation_matrix = correlation_matrix[high_corr_cols].loc[high_corr_cols]
        summary['correlation'] = correlation_matrix

    # Features with high missing values
    high_missing = df.isnull().mean() * 100
    high_missing_features = high_missing[high_missing > 50].sort_values(ascending=False)
    if not high_missing_features.empty:
        summary['high_missing'] = high_missing_features

    return summary

def get_columns_for_analysis(summary):
    # Ask LLM to suggest columns for histogram, bar chart, and pairplot based on dataset summary.
    functions  = [{
        "name": "get_columns_for_analysis",
        "description": "From the given summary, suggest good columns for histogram, bar chart and pairplot",
        "parameters":{
            "type":"object",
            "properties": {
                "histogram": {"type": "string","description": "Column to be chosen for histogram"},
                "barchart": {"type": "string","description": "Column to be chosen for barchart"},
                "pairplot1": {"type": "string","description": "First column to be chosen for pairplot"},
                "pairplot2": {"type": "string","description": "Second column to be chosen for pairplot"},
            },
            "required":["histogram","barchart","pairplot1","pairplot2"]
        }
    }]
    prompt = f'''Given the dataset summary, suggest a numeric column for histogram, 
                 a categorical column for barchart, and two correlated numeric columns for pairplot.
                \nSummary:{summary}'''
    
    json_data = {"model": Config.MODEL,
                "functions": functions,
                "function_call": {"name": "get_columns_for_analysis"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(Config.URL, headers=Config.HEADERS, json=json_data)
    result = response.json()
    print("Token usage till column suggestion: ", result.get('monthlyCost', 'N/A'))
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def encode_image(image_path):
  # Function to encode the image
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def generate_readme(summary, visualizations):
    # Ask LLM to generate README.md content based on dataset summary and visualizations.

    # Encode images to base64
    encoded_images = []
    for image_path in visualizations:
        base64_image = encode_image(image_path)
        encoded_images.append(f"data:image/jpeg;base64,{base64_image}")

    image_filenames = [os.path.basename(f) for f in visualizations]

    functions  = [{
        "name": "generate_readme",
        "description": "Write a story describing the dataset, analysis, insights, and implications.",
        "parameters":{
            "type":"object",
            "properties": {
                "text": {"type": "string","description": "Analysis of the dataset."},
            },
            "required":["text"]
        }
    }]
    prompt = f'''Given the dataset summary: {summary} and visualizations, 
                write a README.md containing dataset's purpose, key findings,
                deeper insights referencing relevant visualizations, and actionable recommendations. 
                Use placeholders {image_filenames} for images.'''
    
    json_data = {"model": Config.MODEL,
                "functions": functions,
                "function_call": {"name": "generate_readme"},
                "messages": [{"role": "user", "content": prompt}] 
                          + [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]} for image in encoded_images]
                }
    response = requests.post(Config.URL, headers=Config.HEADERS, json=json_data)
    result = response.json()
    print("Total token usage by the end of analysis: ", result.get('monthlyCost', 'N/A'))
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def create_output_folder(folder_name):
    # Create an output folder if it doesn't exist.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def create_graphs(df,good_columns,output_folder):
    # Generate and save histogram, bar chart, and scatterplot based on specified columns.
    hist_col = good_columns["histogram"]
    bar_col = good_columns["barchart"]
    pair_cols=(good_columns["pairplot1"],good_columns["pairplot2"])

    # Create histogram
    plt.figure(figsize=(8, 6))
    df[hist_col].dropna().hist(bins=30, color='skyblue', edgecolor='black', label=f'Frequency of {hist_col}')
    plt.title(f'Distribution of {hist_col}')
    plt.xlabel(hist_col)
    plt.ylabel('Frequency')
    plt.legend(title="Legend", loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{hist_col.replace(' ', '_')}_distribution.png")
    plt.close()

    # Create barchart
    value_counts = df[bar_col].value_counts()
    top_values = value_counts[:10] if len(value_counts) > 10 else value_counts

    plt.figure(figsize=(8, 6))
    top_values.plot(kind='bar', color='skyblue', edgecolor='black',label=f'Counts of {bar_col}')
    plt.title(f'Top Categories of {bar_col}')
    plt.xlabel(bar_col)
    plt.ylabel('Counts')
    plt.legend(title="Legend", loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{bar_col.replace(' ', '_')}_barplot.png")
    plt.close()

    # Create scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(df[pair_cols[0]], df[pair_cols[1]], color='skyblue', edgecolor='black', alpha=0.7, label=f'{pair_cols[0]} vs {pair_cols[1]}')
    plt.title(f'Scatterplot of {pair_cols[0]} vs {pair_cols[1]}')
    plt.xlabel(pair_cols[0])
    plt.ylabel(pair_cols[1])
    plt.legend(title="Legend", loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{pair_cols[0].replace(' ', '_')}_vs_{pair_cols[1].replace(' ', '_')}_scatterplot.png")
    plt.close()


def main():
    # Main entry point for the script.
    # Validate command line argument count
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <input_filename.csv>")
        sys.exit(1)

    file = sys.argv[1]
    if not os.path.exists(file):
        print("File not found in the current working directory.")
        sys.exit(1)

    # Create output folder
    file_base_name = os.path.splitext(os.path.basename(file))[0]
    output_folder = create_output_folder(file_base_name)

    # Load and summarize data
    df = pd.read_csv(file, encoding='ISO-8859-1')
    summary = summarize_data(df)

    # Get columns for analysis
    good_columns = get_columns_for_analysis(summary)
    
    # Create visualizations
    create_graphs(df,good_columns,output_folder)

    # Resize the generated images to 512 x 512 pixels
    for imagename in os.listdir(output_folder):
        if imagename.endswith('.png'):
            image_path = os.path.join(output_folder, imagename)
            img = Image.open(image_path)
            img_resized = img.resize((512, 512))
            img_resized.save(image_path)
    
    # Generate README.md
    visualizations = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')]
    story = generate_readme(summary, visualizations)['text']

    with open(os.path.join(output_folder, "README.md"), "w") as readme_file:
        readme_file.write(f"# Automated Analysis of {file}\n\n")
        readme_file.write(story)

    print("Analysis complete. README.md saved successfully.")

main()
