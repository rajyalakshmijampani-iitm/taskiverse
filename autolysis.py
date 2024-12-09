# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "requests"
# ]
# ///

import os
import sys
import json
import pandas as pd
import requests
import matplotlib
import seaborn

# ----------------------------Global variables-----------------------------------------------------

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")                                # Read the token from environment
url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"      # API Endpoint
headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
          }
model = "gpt-4o-mini"
all_functions = {}

#-----------------------------------------------------------------------------------------------------

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
                For numerical columns, treat values in scientific notation (e.g., 9.78E+12) as integers.
                If a column contains all null values, classify its data type as null.
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

def clean_and_analyze_data(df):
    summary={}
    #Basic data cleaning: delete duplicate rows, empty rows
    df = df.drop_duplicates()
    df = df.dropna(how='all')

    #Numeric data summary
    numeric_summary = df.describe().transpose() 
    numeric_missing = df.isnull().mean() * 100  # Find Missing percentage
    numeric_summary['missing%'] = numeric_missing[numeric_summary.index]

    #Categorical data summary
    categorical_summary = df.describe(include='object').transpose()
    categorical_missing = df.isnull().mean() * 100  # Find Missing percentage
    categorical_summary['missing%'] = categorical_missing[categorical_summary.index]

    summary['numeric'] = numeric_summary
    summary['categorical'] = categorical_summary
    return summary

def get_visuals(metadata, summary):
    global all_functions
    all_functions['get_visuals']  = [{"name": "get_visuals",
                                      "description": "From the given metadata and summary statistics of a CSV dataset, generate three best visualizations. The function decides on the best visualizations (such as histograms, bar charts, box plots, scatter plots, or combinations of multiple columns) for the given data, and generates the corresponding Python code to create those visualizations using matplotlib and seaborn. The function outputs a dictionary with the details of each figure, including the file name to save the figure and the code to generate it.",
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
                For each visualization, recommend the most appropriate graph based on the columns' data types and relationships.
                If necessary, generate combined visualizations that explore the relationships between multiple columns.
                Provide the corresponding file name to store the figure and Python code using `matplotlib` and `seaborn` libraries only to generate the suggested visualizations. 
                The dataframe is stored in a variable named `df`.
                Ensure that the code is error-free, ready to execute, and does not include any escape sequences or comments.
                \n\nMetadata:\n{metadata}\n\nSummary Statistics:\n{summary}'''
    
    json_data = {"model": model,
                "functions": all_functions['get_visuals'],
                "function_call": {"name": "get_visuals"},
                "messages":[{"role": "user", "content": prompt}]
                }
    response = requests.post(url, headers=headers, json=json_data)
    result = response.json()
    return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])


def create_output_folder(folder_name):
    """Create an output folder for the given dataset name."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def main():

    # Validate command line argument count
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <input_filename.csv>")
        sys.exit(1)

    # Check file existence in current working directory
    file = sys.argv[1]
    if not os.path.exists(file):
        print("File not found in the current working directory.")
        sys.exit(1)

    # Create a folder with file base name to store results
    file_base_name = os.path.splitext(os.path.basename(file))[0]
    output_folder = create_output_folder(file_base_name)

    # Load a sample of 10 lines (or all lines if the file has fewer than 10 lines) from the file
    try:
        with open(file=file, mode='r', encoding='utf-8') as f:
            line_count = int(sum(1 for _ in f))
            if line_count == 0:
                raise ValueError(f"The file '{file}' is empty.")
            f.seek(0)
            data = ''.join([f.readline() for _ in range(min(10,line_count))])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except:
        print(f"Error: There was an issue reading the file '{file}'.")
        sys.exit(1)

    # Send the sample data to LLM and get metadata
    metadata = get_metadata(data) 

    # Load, clean and perform basic analysis on the CSV
    df = pd.read_csv(file)
    summary = clean_and_analyze_data(df)

    #From metadata + summary statistics, get code for 3 best visualizations
    visuals = get_visuals(metadata,summary)




if __name__ == "__main__":
     main()
