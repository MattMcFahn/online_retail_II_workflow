"""
 Helper script for the retail data pipeline.
 
 Contains a collection of helper functions that aren't specific to the problem,
 but provide generic functionality useful in the analysis.
 
 @author: Matt McFahn
"""

import re
from pandas import ExcelWriter

### - Formatting helpers
def __title__(string):
    """
    str -> str
    Converts a string to title case from snake case
    Used to help clean formatting for the pptx

    Examples:
        - __title__(coastal_inland) = Coastal Inland
    """
    return string.replace('_',' ').title()

def __minus_bad_chars(string):
    """
    Replaces any instance of a bad char with a _ (to avoid bad names for windows
    files or excel sheetnames).

    Examples:
        __minus_bad_chars('Hello?')='Hello_'
    """
    # Characters not allowed to replace in regex class
    bad_chars = '''[/?<>\:*|"]'''
    # Run regex replacement
    good_string = re.sub(bad_chars, '_',string)
    return good_string

### - Output helpers
def dictionary_dump(frames, outputs, filename):
    """
    dict, str, str -> None
    
    Outputs a dictionary of pandas dataframes as an excel workbook in the 
    outputs location (with each sheet having a key name from the dict).
    
    Will raise an error if one of the values in the dict isn't a dataframe
    """
    out_path = fr'{outputs}\{filename}.xlsx'
    with ExcelWriter(out_path) as writer:
        for key, frame in frames.items():
            written_key = __title__(__minus_bad_chars(key))
            frame.to_excel(writer, sheet_name = written_key)
    return None
