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
from PIL import Image
import warnings

# Suppress warnings globally
warnings.filterwarnings("ignore")
matplotlib.use('Agg')

#Global variables

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")                                # Read the token from environment
if  AIPROXY_TOKEN is None:
    print("Please set token to proceed")
    sys.exit(1)

url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"      # API Endpoint
headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
          }
model = "gpt-4o-mini"
df = None
output_folder=None

def summarize_data(df):
    # Basic data cleaning: delete duplicate rows, empty rows
    summary={}
    
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    missing = df.isnull().mean() * 100
    unique = df.nunique()

    # Numeric data summary
    numeric_summary = df.describe().transpose()
    numeric_summary['missing%'] = missing[numeric_summary.index]
    numeric_summary['unique_values'] = unique[numeric_summary.index]

    # Categorical data summary
    categorical_summary = df.describe(include='object').transpose()
    categorical_summary['missing%'] = missing[categorical_summary.index]
    categorical_summary['unique_values'] = unique[categorical_summary.index]

    summary['numeric'] = numeric_summary
    summary['categorical'] = categorical_summary
    return summary

def get_columns_for_analysis(summary):
    functions  = [{"name": "get_columns_for_analysis",
                   "description": "From the given summary, suggest good columns for histogram, bar chart and pairplot",
                    "parameters":{"type":"object",
                                    "properties": {"histogram": {"type": "string",
                                                                "description": "Column to be chosen for histogram"
                                                        },
                                                    "barchart": {"type": "string",
                                                                "description": "Column to be chosen for barchart"
                                                        },
                                                    "pairplot1": {"type": "string",
                                                                "description": "First column to be chosen for pairplot"
                                                        },
                                                    "pairplot2": {"type": "string",
                                                                "description": "Second column to be chosen for pairplot"
                                                        },
                                    
                                    },
                                    "required":["histogram","barchart","pairplot1","pairplot2"]
                                                    }
                                        }]
    prompt = f'''Given the dataset summary, suggest one good column to be chosen for histogram,
                one good column to be chosen for barchart, two good columns to be chosen for pairplot.
                \nSummary:{summary}'''
    
    json_data = {"model": model,
                "functions": functions,
                "function_call": {"name": "get_columns_for_analysis"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    print("Total token usage till now: ",result['monthlyCost'])
    print(result)
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])


def generate_readme(summary, visualizations):
    functions  = [{"name": "generate_readme",
                    "description": "Write a story describing the dataset, analysis, insights, and implications.",
                    "parameters":{"type":"object",
                                  "properties": {"text": {"type": "string",
                                                          "description": "Analysis of the dataset."
                                                        },
                                                },
                                  "required":["text"]
                                 }
                }]
    prompt = f'''Given the dataset summary: {summary} and visualizations: {visualizations}, 
                write a README.md containing dataset's purpose, key findings, insights, and recommendations. 
                Use placeholders to indicate where the visualization would be integrated.
                Explain how each visualization supports the narrative.'''
    
    json_data = {"model": model,
                "functions": functions,
                "function_call": {"name": "generate_readme"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    print("Total token usage till now: ",result['monthlyCost'])
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def create_output_folder(folder_name):
    # Create an output folder for the given dataset name.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def create_graphs(df,good_columns):

    global output_folder

    hist_col = good_columns["histogram"]
    bar_col = good_columns["barchart"]
    pair_cols=(good_columns["pairplot1"],good_columns["pairplot2"])

    # Create histogram
    plt.figure(figsize=(8, 6))
    df[hist_col].dropna().hist(bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {hist_col}')
    plt.xlabel(hist_col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{hist_col}_distribution.png")
    plt.close()

    # Create barchart
    top_n = 10  # Number of top categories to display
    value_counts = df[bar_col].value_counts()
    # Handle value_counts for better visualization
    if len(value_counts) > top_n:
        top_values = value_counts[:top_n]
    else:
        top_values = value_counts
    # Plot the optimized bar chart
    plt.figure(figsize=(8, 6))
    top_values.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Top {top_n} Categories of {bar_col}')
    plt.xlabel(bar_col)
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{bar_col}_barplot.png")
    plt.close()

    # Create pairplot
    plt.figure(figsize=(8, 6))
    plt.scatter(df[pair_cols[0]], df[pair_cols[1]], color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Scatterplot of {pair_cols[0]} vs {pair_cols[1]}')
    plt.xlabel(pair_cols[0])
    plt.ylabel(pair_cols[1])
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{pair_cols[0]}_vs_{pair_cols[1]}_scatterplot.png")
    plt.close()


def main():
    global output_folder
    global df
    # Validate command line argument count
    print("Validating input...")
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <input_filename.csv>")
        sys.exit(1)

    # Check file existence in current working directory
    print("Checking for file existence...")
    file = sys.argv[1]
    if not os.path.exists(file):
        print("File not found in the current working directory.")
        sys.exit(1)

    # Create a folder with file base name to store results
    print("Creating output folder...")
    file_base_name = os.path.splitext(os.path.basename(file))[0]
    output_folder = create_output_folder(file_base_name)

    # Load, clean and perform basic analysis on the CSV
    print("Summarizing the data...")
    df = pd.read_csv(file, encoding='ISO-8859-1')
    summary = summarize_data(df)

    # Ask LLM what columns to analyze
    print("Getting LLM response on what columns to analyze...")
    good_columns = get_columns_for_analysis(summary)
    
    # Create the graphs
    print("Generating graphs...")
    create_graphs(df,good_columns)

    print("Resizing the images...")
    # Resize the generated images to 512 x 512 pixels
    for imagename in os.listdir(output_folder):
        if imagename.endswith('.png'):
            image_path = os.path.join(output_folder, imagename)
            img = Image.open(image_path)
            img_resized = img.resize((512, 512))
            img_resized.save(image_path)
    
    print("Generating README...")
    # From summary and visualizations, generate README.md
    visualizations = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    story = generate_readme(summary, visualizations)['text']

    readme_path = os.path.join(output_folder, "README.md")

    with open(readme_path, "w") as readme_file:
        readme_file.write(f"# Automated Analysis of {file}\n\n")
        readme_file.write(story)

    print("Analysis complete. README.md saved successfully.")


main()
