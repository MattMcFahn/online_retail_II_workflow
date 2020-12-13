"""
 Main functionality for the data workflow for select insight extraction from 
 Kaggle's "online_retail_II" data: www.kaggle.com/mashlyn/online-retail-ii-uci
 
 General steps performed:
     > Profiling
     > Cleaning / Munging / Wrangling
     > Modelling
         * k-means & factor analysis of customers and products
     > Validation
     > Output (dataframes for reporting)
 
 A tableau dashboard has been set up for reporting:
     TODO: Add link to dashboard
 
 @author: Matt McFahn
"""

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import retail_helpers
import retail_cleaning

# Dataset location (replace with the location of the online_retail_II csv file)
source = r'C:\Users\mattm\Documents\Coding\Tableau_Data_Sources\online_retail_II.csv'

# Repository location (self explanatory: local path)
repo = r'C:\Users\mattm\Documents\GitHub\online_retail_II_workflow'

# Output locations: profiling & outputs. For descriptive outputs, and reporting outputs
profiling = rf'{repo}\Profiling'
outputs = rf'{repo}\Outputs'

def __init(**files):
    """
    Simple initial helper to catch non-existence of filepaths used. If any don't
    exist, it will prompt the user to confirm whether they want them to be created.
    """
    # TODO: using os module (not necessary for non-distributed use)
    return None

def __load_data(source):
    """ 
    str -> pd.DataFrame
    
    Load the data from source, return as pandas dataframe.
    Also performs some standard profiling of the given data 
    (distributional and point estimates).
    """
    # Specify datatypes for InvoiceDate as it loads as generic object
    dataframe = pd.read_csv(source, parse_dates = ['InvoiceDate'])
    
    return dataframe

def __simple_profiling(dataframe, profiling):
    """
    pd.DataFrame, str -> dict(pd.DataFrame)
    
    Performs some simple profiling of the source data, outputting to the 
    "profiling" location on the local system (and returning the profiles).
    """
    # Initialise dictionary of outputs to add to
    frames = {}
    
    # Print general info
    dataframe.info()
    
    # Return a random sample to explore visually (alternative to pd.DataFrame.head)
    sample_df = dataframe.sample(200)
    frames['sample'] = sample_df
    
    # Descriptive stats for numeric cols
    descriptive_df = dataframe.describe()
    frames['description'] = descriptive_df
    retail_helpers.dictionary_dump(frames = frames, 
                                   outputs = outputs, 
                                   filename = 'Descriptive')
    
    return None

def main():
    """
    Full pipeline for data processing, modelling, and validation.
    """
    # Load data
    dataframe = __load_data(source)
    __simple_profiling(dataframe, profiling)
    
    # Run all cleaning
    df_cleaned = retail_cleaning.cleaning_main(dataframe)
    
    # Run modelling
    # TODO
    
    # Post modelling wrangling
    
    # Output to local locations
    
    return None

