# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "requests",
#     "Pillow"
# ]
# ///

import os
import sys
import json
import pandas as pd
import requests
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Suppress warnings globally
warnings.filterwarnings("ignore")

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
all_functions = {}
df = None
output_folder=None

def get_metadata(data):
    # Expected output format
    # {
    #     "columns": [{"name":"col1","type":"string"},
    #                 {"name":"col2","type":"integer"},
    #                  ...
    #                ]
    # }

    global all_functions
    all_functions['extract_metadata']  = [{"name": "extract_metadata",
                                            "description": "From the provided CSV data, extract the column names and their corresponding data types.",
                                            "parameters":{"type":"object",
                                                            "properties": {"columns": {"type": "array",
                                                                                    "items":{"type": "object",
                                                                                                "description": "Details of column",
                                                                                                "properties": {"name": {"type": "string",
                                                                                                                        "description": "Name of the column"
                                                                                                                        },
                                                                                                                "type": {"type": "string",
                                                                                                                        "description": "Datatype of the column (e.g. integer, string)"
                                                                                                                        }
                                                                                                            },
                                                                                                "required": ["name", "type"]
                                                                                                }   
                                                                                    }
                                                                            },
                                                            "required":["columns"]
                                                            }
                                            }]
    
    prompt = '''You are a data analysis assistant with a focus on inferring the appropriate data types for each column in a CSV file.
                From the provided CSV data, extract the column names and infer their corresponding data types.
                The data types should be one of the following: string, float, integer, boolean, or datetime.
                Inference should be based not only on the data but also on the column names, as some columns may contain unclean or irrelevant data.
                The goal is to provide the most appropriate data type for each column based on its name and the overall content.'''
    
    json_data = {"model": model,
                 "functions": all_functions['extract_metadata'],
                 "function_call": {"name": "extract_metadata"},
                 "messages":[{"role": "system", "content": prompt},
                             {"role": "user", "content": data}
                            ]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

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

def get_code_for_cleaning(metadata,summary):
    global all_functions
    all_functions['code_for_cleaning']  = [{"name": "code_for_cleaning",
                                            "description": "From the given metadata and summary statistics, generate Python code to clean the dataset.",
                                            "parameters":{"type":"object",
                                                          "properties": {"code": {"type": "string",
                                                                                "description": "Python code to clean the dataset."
                                                                                }
                                                                        },
                                                        "required": ["code"]
                                                    }
                                            }]
                                                                            
    prompt = f'''Given the following metadata and summary statistics, provide Python code to clean the dataset.
                The dataframe is stored in a variable named `df`. The cleaned data should also remain in the variable `df`.
                Use only pandas library.
                Do not include any escape sequences for newline character or quotes.
                Do not include comments.
                Metadata:\n{metadata}\nSummary Statistics:\n{summary}'''
    
    json_data = {"model": model,
                "functions": all_functions['code_for_cleaning'],
                "function_call": {"name": "code_for_cleaning"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def re_request_code(prev_code,error):
    global all_functions
    all_functions['re_request_code']  = [{"name": "re_request_code",
                                        "description": "Given the code and error message, provide a new code to avoid the error.",
                                        "parameters":{"type":"object",
                                                        "properties": {"code": {"type": "string",
                                                                                "description": "Updated code to avoid the given error."
                                                                                }
                                                                    },
                                                        "required": ["code"]
                                                    }
                                            }]
                                                                            
    prompt = f'''The following code caused an error. Please analyze the issue and provide a corrected version of the code.
                    Use only pandas library.
                    Do not include any escape sequences for newline character or quotes.
                    Do not include comments.
                \n\Code:\n{prev_code}\n\nError:\n{error}'''
    
    json_data = {"model": model,
                "functions": all_functions['re_request_code'],
                "function_call": {"name": "re_request_code"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def get_visuals(metadata, summary):
    global all_functions
    all_functions['get_visuals']  = [{"name": "get_visuals",
                                      "description": "From the given metadata and summary statistics of a CSV dataset, generate code for three best visualizations.",
                                       "parameters":{"type":"object",
                                                     "properties": {"fig1": {"type": "object",
                                                                              "description": "Details of Figure 1, including the file name and the Python code to generate it.",
                                                                              "properties": {"file_name": {"type": "string",
                                                                                                           "description": "The appropriate file name for saving Figure 1."
                                                                                                           },
                                                                                              "code": {"type": "string",
                                                                                                       "description": "Python code to generate Figure 1 using matplotlib and seaborn."
                                                                                                      }
                                                                                            },
                                                                              "required": ["file_name", "code"]
                                                                            },
                                                                    "fig2": {"type": "object",
                                                                              "description": "Details of Figure 2, including the file name and the Python code to generate it.",
                                                                              "properties": {"file_name": {"type": "string",
                                                                                                           "description": "The appropriate file name for saving Figure 2."
                                                                                                           },
                                                                                              "code": {"type": "string",
                                                                                                       "description": "Python code to generate Figure 2 using matplotlib and seaborn."
                                                                                                      }
                                                                                            },
                                                                              "required": ["file_name", "code"]
                                                                            },
                                                                    "fig3": {"type": "object",
                                                                              "description": "Details of Figure 3, including the file name and the Python code to generate it.",
                                                                              "properties": {"file_name": {"type": "string",
                                                                                                           "description": "The appropriate file name for saving Figure 3."
                                                                                                           },
                                                                                              "code": {"type": "string",
                                                                                                       "description": "Python code to generate Figure 2 using matplotlib and seaborn"
                                                                                                      }
                                                                                            },
                                                                              "required": ["file_name", "code"]
                                                                            }
                                                                    },
                                                      "required":["fig1","fig2","fig3"]
                                                    }
                                    }]
    prompt = f'''Given the following metadata and summary statistics of the CSV data, analyze the columns and identify three best visualizations for exploration.
                Provide file name to store the figure and the Python code.
                Use only `matplotlib` and `seaborn` libraries. 
                The folder name is in `output_folder` variable. Use os.path.join(output_folder, filename) to save.
                The dataframe is stored in a variable named `df`.
                Do not include any escape sequences for newline character or quotes.
                Do not include comments.
                \n\nMetadata:\n{metadata}\n\nSummary Statistics:\n{summary}'''
    
    json_data = {"model": model,
                "functions": all_functions['get_visuals'],
                "function_call": {"name": "get_visuals"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

def generate_readme(summary, visualizations):
    global all_functions
    all_functions['generate_readme']  = [{"name": "generate_readme",
                                      "description": "From the given summary and the visualizations generated, Write a story describing the dataset, analysis, insights, and implications.",
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
                Integrate the visualizations at relevant points using placeholders.
                Ensure that they align with the insights. Explain how each visualization supports the narrative.'''
    
    json_data = {"model": model,
                "functions": all_functions['generate_readme'],
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

def execute_code(code):
    try:
        exec(code)
        execution_status = 'Success'
    except Exception as e:
        error_message = str(e)
        print(error_message)
        execution_status = 'Failure'

    # Re-request code
    if execution_status == 'Failure':
        try:
            updated_code = re_request_code(code,error_message)['code']
            exec(updated_code)
            execution_status = 'Success'
        except Exception as e:
            error_message = str(e)
            print(error_message)
            print('Invalid code received from the LLM. The process was halted after 2 unsuccessful attempts.')
            execution_status = 'Failure'
    
    return execution_status

def create_graphs(df):
    global output_folder
    # Creates graphs for the first numerical and categorical columns of a DataFrame.

    # Find the first numerical column
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        num_col = numerical_cols[0]
        plt.figure(figsize=(8, 6))
        df[num_col].dropna().hist(bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {num_col}')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{num_col}_distribution.png")
        plt.close()
    else:
        print("No numerical columns found in the DataFrame.")

    # Find the first categorical column
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_col = categorical_cols[0]
        plt.figure(figsize=(8, 6))
        df[cat_col].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title(f'Counts of {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel('Counts')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{cat_col}_barplot.png")
        plt.close()
    else:
        print("No categorical columns found in the DataFrame.")

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

    # Load a sample of 10 lines (or all lines if the file has fewer than 10 lines) from the file
    print("Sampling data from the csv...")
    try:
        with open(file=file, mode='r', encoding='ISO-8859-1') as f:
            line_count = int(sum(1 for _ in f))
            if line_count == 0:
                raise ValueError(f"The file '{file}' is empty.")
            f.seek(0)
            data = ''.join([f.readline() for _ in range(min(10,line_count))])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except:
        print(f"Error: There was an issue reading the file '{file}'. Try different encoding format..")
        sys.exit(1)

    # Send the sample data to LLM and get metadata
    print("Getting metadata from LLM...")
    try:
        metadata = get_metadata(data)
    except:
        metadata = None

    # Load, clean and perform basic analysis on the CSV
    print("Summarizing the data...")
    df = pd.read_csv(file, encoding='ISO-8859-1')
    summary = summarize_data(df)

    # From metadata + summary statistics, get code for data cleaning
    print("Getting code for data cleaning from LLM...")
    try:
        code_for_cleaning = get_code_for_cleaning(metadata,summary)
    except:
        code_for_cleaning = None
    
    # Perform data cleaning with the obtained code
    print("Cleaning the code...")
    if code_for_cleaning:
        cleaning_status = execute_code(code_for_cleaning['code'])
        if cleaning_status=='Success':
            summary = summarize_data(df)  # Updated data summary after data clean up
        else:
            print('LLM failed twice in giving code for data cleaning. Continuing with raw data.')

    # From metadata + summary statistics, get code for 3 best visualizations
    print("Getting code for visualizations...")
    visuals = get_visuals(metadata,summary)

    no_of_figs_generated = 0

    print("Creating the visuals....")
    for fig in ['fig1','fig2','fig3']:
        fig_generation_status=execute_code(visuals[fig]['code'])
        if fig_generation_status == 'Success':
            no_of_figs_generated+=1
    
    if no_of_figs_generated == 0:   # LLM Code failed for all 3 figures
        print('LLM failed twice in generating figures for analysis. Generating from autolysis.py.')
        create_graphs(df,output_folder)

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
if __name__ == "__main__":
     main()
