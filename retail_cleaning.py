"""
 A script for the initial data wrangling and cleaning processes.
 
 Fuzzy logic using the Levenshtein distance is employed to match "descriptions"
 that are actually recording issues with orders. The results of this are used
 for a more manual classification of typical words that correspond to similar 
 issues.
 
 This manual classification is then used with regex matching to create typical
 classifications of issues with orders (e.g. "Damanged/Faulty", "Missing",
 "Wrong Item", etc.)
 
  Main function: "cleaning_main"
 
 @author: Matt McFahn
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
import re

def __matching_descriptions_helper(nonupper):
    """
    pd.DataFrame -> pd.DataFrame
    
    USED ONLY AS A HELPER AT PRESENT
    
    Pass a dataframe of items with duplicate descriptions, and identify similar
    descriptions using Levenshtein distance. This information can be used with
    human judgement to provide a cleaned set of descriptions, mapping from old
    
    Returns: A dataframe matching old descriptions to suggested categories,
    as well as similarity scores
    
    Looking at the "similarity" dataframe helps speed up a more manual strategy 
    (employed in the function below)
    """
    df = nonupper.copy()
    
    unique_descr = df['Description'].unique()
    
    score_sort = [(x,) + i
                  for x in unique_descr 
                  for i in process.extract(x, unique_descr, scorer=fuzz.token_sort_ratio)]
    similarity = pd.DataFrame(score_sort, columns=['description','potential_match','score_sort'])
    similarity['sorted_description'] = np.minimum(similarity['description'], similarity['potential_match'])
    
    suggestions = similarity.loc[(similarity['score_sort'] >= 70) &
                                 ~(similarity['description'] == similarity['potential_match']) &
                                 ~(similarity['description'] == similarity['sorted_description'])]
    suggestions.drop(columns = {'sorted_description'}, inplace = True)
    
    totals = suggestions.groupby(['description','score_sort']).agg(
                                {'potential_match': ', '.join}).sort_values(
                                ['score_sort'], ascending=False).reset_index()
    
    return totals

def __map_item_issues_to_groupings(df):
    """
    pd.DataFrame -> pd.DataFrame
    
    Based on a quick exploration of the Levenshtein matching employed above, 
    defines a set of typical order issues they've referred to, and words within
    the description(s) that will indicate whether that description is referring
    to that issue
    
    This is a simple method that provides a first pass at classifying these
    issues. A more sophisticated method could be employed, but it would likely
    be a difficult issue to solve. Introducing fields dedicated to recording
    order issues in data collection is necessary for future data recording
    """
    # Define the issue categories, and regex matching words that indicates groupings
    issue_categories = {'Damaged':['damage','dirty','throw','wet','discolour','faulty','mouldy','unsale','throw','crush','crack','broke','damges','rust'],
                        'Wrong Item':['credit','incorrect','wrong'],
                        'Online (?)':['dotcom','ebay','amazon'],
                        'Wrong Entry':['entry','error'],
                        'Missing from order':['lost','missing'],
                        'Mix up':['wrongly','mix up']}
    # Final category will be 'Unclassified issue'. We could classify more, but
    # it would be too time intensive for this exercise
    
    df['IssueCategory'] = 'Unclassified'
    df['Description_Lower'] = df['Description'].str.lower()
    for classification, term_list in issue_categories.items():
        for term in term_list:
            df.loc[df['Description_Lower'].str.contains(term), 'IssueCategory'] = classification
    
    df['IssueWithItem'] = True
    
    # Show number in each group, just for some simple reporting. Lot of unclassified still!
    df.groupby(['IssueCategory'])['Description'].count()
    
    return df[['Description','IssueWithItem','IssueCategory']]

def __overwrite_check_records(dataframe, duplicates):
    """
    pd.DataFrame, pd.DataFrame -> pd.DataFrame, pd.DataFrame
    
    Records where the description is "check" get overwritten with the other 
    description matching that StockCode. duplicates is adjusted accoringly
    """
    top_descriptions = duplicates.groupby('StockCode')['Quantity'].max().reset_index()
    top_descriptions = top_descriptions.merge(duplicates, on = ['StockCode','Quantity'],
                                              how = 'inner', validate = 'one_to_many')
    
    # Identify any duplicate StockCodes to deal with manually
    #dupes = top_descriptions.loc[top_descriptions.duplicated(subset = 'StockCode', keep = False)]
    
    # Update top descriptions manually
    top_descriptions = top_descriptions.loc[~(top_descriptions['Description'] == 'S/16 VINTAGE IVORY CUTLERY')] # Cases where a StockCode has two max Quantities
    top_descriptions = top_descriptions.loc[~((top_descriptions['Description'] == 'found') & (top_descriptions['StockCode'] == '35598C'))] 
    
    top_descriptions.reset_index(inplace = True, drop = True)
    top_descriptions.rename(columns = {'Description':'Description_New'}, inplace = True)
    top_descriptions = top_descriptions[['StockCode','Description_New']]
    
    # Identify "check" records
    checks = duplicates.loc[duplicates['Description'].str.contains('check')]
    checks = checks.merge(top_descriptions, on = 'StockCode', validate = 'one_to_one')
    checks.drop(columns = {'Quantity'}, inplace = True)
    
    # Update dataframe
    dataframe = dataframe.merge(checks, on = ['StockCode', 'Description'], how = 'left')
    dataframe.loc[~(dataframe['Description_New'].isna()), 'Description'] = dataframe.loc[~(dataframe['Description_New'].isna()), 'Description_New']
    dataframe.drop(columns = {'Description_New'}, inplace = True)
    
    # Update duplicates
    duplicates = duplicates.merge(checks, on = ['StockCode', 'Description'], how = 'left')
    duplicates.loc[~(duplicates['Description_New'].isna()), 'Description'] = duplicates.loc[~(duplicates['Description_New'].isna()), 'Description_New']
    duplicates.drop(columns = {'Description_New'}, inplace = True)
    duplicates = duplicates.groupby(['StockCode','Description'])['Quantity'].sum().reset_index()
    duplicates = duplicates.loc[duplicates.duplicated(subset = 'StockCode', keep = False)]
    
    return dataframe, duplicates

def __overwrite_duped_descriptions(dataframe, duplicates):
    """
    pd.DataFrame, pd.DataFrame -> pd.DataFrame, pd.DataFrame
    
    Finds where the description is some kind of "issue", and overwrites by 
    merging with StockCode and taking the other entry. This is after a flag has
    already been added to record the "issues" found in the Description field
    """
    # Create a mapping of "issue" descriptions to new descriptions
    new_descr = duplicates.groupby(['StockCode'])['Quantity'].max().reset_index()
    new_descr = new_descr.merge(duplicates, on = ['StockCode','Quantity']).rename(columns = {'Description':'Description_New','Quantity':'QuantityMax'})
    new_descr = new_descr.loc[~(new_descr['Description_New'] == 'S/16 VINTAGE IVORY CUTLERY')] # Cases where a StockCode has two max Quantities
    new_descr = new_descr.loc[~((new_descr['Description_New'] == 'found') & (new_descr['StockCode'] == '35598C'))] 
    
    descr_mapping = new_descr.merge(duplicates, on = ['StockCode'])
    descr_mapping.drop(columns = {'Quantity','QuantityMax'}, inplace = True)
    
    # Use this to update Descriptions in the main dataframe
    dataframe = dataframe.merge(descr_mapping, on = ['StockCode', 'Description'], how = 'left')
    dataframe.loc[~(dataframe['Description_New'].isna()), 'Description'] = dataframe.loc[~(dataframe['Description_New'].isna()), 'Description_New']
    dataframe.drop(columns = {'Description_New'}, inplace = True)
    
    return dataframe
    
def __address_duplicate_descriptions(dataframe):
    """
    pd.DataFrame -> pd.DataFrame
    
    There are a number of instances of a StockCode having >1 unique description.
    
    Primarily, the non UPPER CASE descriptions refer to some kind of 'Issue'
    with that item in that order.
    
    This function addresses these by the following:
        * Any saying 'check' are overwritten by the correct description 
          (i.e. missing description)
        * Other 'descriptions' that aren't upper case are classified as typical
        issue types, and;
    
    Additional fields are added to the dataframe to indicate which items had 
    issues in their orders, and what type of issue it was.
    """
    # Some common cases where only a couple of letters are lower case
    dataframe['Description'] = dataframe['Description'].str.replace('x40cm', 'X40CM')
    dataframe['Description'] = dataframe['Description'].str.replace('x45cm', 'X45CM')
    dataframe['Description'] = dataframe['Description'].str.replace('x30CM', 'X30CM')
    dataframe['Description'] = dataframe['Description'].str.replace('x30cm', 'X30CM')
    
    dataframe['Description'] = dataframe['Description'].str.replace('TRADITIONAl', 'TRADITIONAL')
    
    dataframe['Description'] = dataframe['Description'].str.replace('No', 'NO')
    
    # Get unique pairs (item, description), and find duplicates
    items = dataframe.groupby(['StockCode','Description'])['Quantity'].sum().reset_index()
    duplicates = items.loc[items.duplicated(subset = ['StockCode'], keep = False)]
    
    # Overwrite "check" descriptions with the correct labels
    dataframe, duplicates = __overwrite_check_records(dataframe, duplicates)
    
    # Items with "issues" recorded instead of "Description" are non-upper case
    issues = duplicates.loc[~(duplicates.Description.str.isupper())].groupby(['Description'])['Quantity'].sum().reset_index().drop(columns = {'Quantity'})
    # Group these "issue" items according to their issue
    issues = __map_item_issues_to_groupings(issues)
    
    # Add issues & issue categories, overwrite extraneous descriptions
    dataframe = dataframe.merge(issues, on = 'Description', how = 'left')
    dataframe = __overwrite_duped_descriptions(dataframe, duplicates)
    
    # HACK - For StockCode '84968B' which we had to exclude earlier
    dataframe.loc[dataframe['StockCode'] == '84968B', 'Description'] = 'SET OF 16 VINTAGE IVORY CUTLERY'
    
    return dataframe

def cleaning_main(dataframe):
    """
    pd.DataFrame -> pd.DataFrame
    
    Performs two main functions:
        (1) Cleaning:
            * Identify data where 'Description' is missing / inconsistent. 
            Update from records where StockCode matches, AND classify the issues 
            prev. recorded as description.
            
            * Identify records where Quantity unknown. Flag as incomplete data
            
            * Identify records where 'InvoiceDate' missing. Flas as incomplete
            
            * Identify records where price missing. Update using StockCode
            
            * Identify records where Customer ID missing. Flag as incomplete
        
        In addition, overwrite any customer missing values such as "missing"
        with a None datatype.
        
        (2) Extra identification:
            * Add columns to further segment the data / flag incomplete
            (so these can be filtered out of any modelling + reporting)
    
    """
    # Make StockCode upper case
    dataframe['StockCode'] = dataframe['StockCode'].str.upper()
    
    # Drop duplicate entries
    dataframe = dataframe.loc[~(dataframe.duplicated(keep = 'last'))]
    
    dataframe['CancelledOrder'] = dataframe.Invoice.str.contains('C')
    # Create numeric stock codes
    #dataframe['StockCodeNumeric'] = dataframe['StockCode'].apply(lambda x: re.sub(r'[a-zA-z]','', x))
    
    # For string columns, remove trailing and leading spaces (if there)
    for col in dataframe.columns:
        if dataframe[col].dtype == object:
            dataframe[col] = dataframe[col].str.strip()
    
    # Multi descriptions for one Stockcode, overwrite or re-classify as an issue
    dataframe['OrigDescription'] = dataframe['Description']
    dataframe = __address_duplicate_descriptions(dataframe)
    dataframe['IssueWithItem'].fillna(False, inplace = True)
    
    
    # Deal with other incomplete /negative data: price, description, quantity
    dataframe['NoDescriptionOrPrice'] = dataframe['Description'].isna()
    dataframe['PriceIsCredit'] = (dataframe['Price'] < 0)
    dataframe['QuantityLeqZero'] = (dataframe['Quantity'] <= 0)
    
    # Create Time, Date seperately
    dataframe['Date'] = dataframe['InvoiceDate'].dt.date
    dataframe['Time'] = dataframe['InvoiceDate'].dt.time
    
    # Reorder columns
    columns = ['Invoice','StockCode','Description','OrigDescription','Customer ID','Country',
               'Quantity','Price','InvoiceDate','Date','Time',
               'CancelledOrder','IssueWithItem','IssueCategory','QuantityLeqZero','NoDescriptionOrPrice','PriceIsCredit']
    dataframe = dataframe[columns]
    
    return dataframe
