"""
 Modelling functionality for the online_retail_II dataset post cleaning:
     > RFM segmentation of customers
     > Product classifications
     > AOB
 
 This is to be applied to a cleaned dataset. I've formatted the cleaning 
 functionality to only remove duplicates, and split out issues with obvious
 orders via added identifiers. Sub selection of data without issues is still
 needed prior to modelling.
 
 @author: Matt McFahn
"""

import pandas as pd

def __subset_useful_data(df_cleaned):
    """
    pd.DataFrame -> pd.DataFrame, pd.DataFrame
    
    Using identifiers created at the cleaning stage, subset the data to give 
    only the rows we want to use for modelling
    
    Returns two dataframes: one for RFM modelling, one for product modelling
    """
    # Remove cancelled orders
    cut_df = df_cleaned.loc[~df_cleaned['CancelledOrder']]
    
    # RFM modelling: Get rid of NaN customer Id's (22% of data). Nothing else to cut
    rfm_data = cut_df.loc[cut_df['Customer ID'].notna()]
    rfm_data['Customer ID'] = rfm_data['Customer ID'].astype(int)
    
    # Product modelling: Just need StockCode, Description (note multiple StockCodes can have the same description)
    products = cut_df[['StockCode', 'Description']].drop_duplicates().dropna()
    
    return  rfm_data, products

def __RFM_identification(rfm_data):
    """
    pd.DataFrame -> pd.DataFrame
    
    A simple application of Recency, Frequency, Monetary Value scoring for customer
    worth, to identify key customers, and different customer groupings.
    
    Returns a dataframe with columns:
        ["Customer ID", "RecencyScore", "FrequencyScore", "MonetaryScore", "RFMGrouping"]
    
    """
    max_date = rfm_data['Date'].max()
    
    ### - Recency
    recency = (max_date - rfm_data.groupby('Customer ID')['Date'].max()).dt.days
    recency = recency.reset_index()
    recency.rename(columns = {'Date':'DaysSinceLastPurchase(R)'},  inplace = True)
    
    ### - Frequency
    frequency = rfm_data[['Customer ID', 'Date']].drop_duplicates().groupby('Customer ID')['Date'].count().reset_index()
    frequency.rename(columns = {'Date':'NumberOfOrders(F)'},  inplace = True)
    
    ### - Monetary value (assuming Total price = Price * Quantity)
    monetary = rfm_data[['Customer ID', 'Price', 'Quantity']]
    monetary['TotalPrice'] = monetary['Price'] * monetary['Quantity']
    monetary = monetary.groupby('Customer ID')['TotalPrice'].sum().reset_index()
    monetary.rename(columns = {'TotalPrice':'TotalSpent(M)'}, inplace = True)
    
    # - Simple plotting to understand distrns better
    recency.hist(column = 'DaysSinceLastPurchase(R)')
    frequency.hist(column = 'NumberOfOrders(F)')
    monetary.hist(column = 'TotalSpent(M)')
    # Freq & Monetary are power-law, recency exponential. Let's trim freq & mon
    frequency.loc[frequency['NumberOfOrders(F)']<=50].hist(column = 'NumberOfOrders(F)')
    monetary.loc[monetary['TotalSpent(M)']<=10000].hist(column = 'TotalSpent(M)')
    
    
    # Given the power law distrns, we could really spend more time figuring out
    # how to best group our scoring categories. As this is a pilot for now, we
    # go for some simple binning of the customers:
    ## - Recency: Within last 30 days = Recent, ..., Last order over 2 years old = Old customer
    ## - Frequency: >50 orders = Big orderer, ..., 1 Order = One time customer
    ## - Monetary: >100,000 = Huge spender, 10,000-100,000 = Big spender, ..., 0-500 = Small time
    
    # Map frequencies
    recency['Recency'] = pd.cut(recency['DaysSinceLastPurchase(R)'],
                                bins = [-1,30,90,360,800],
                                labels = [4,3,2,1]).astype(int)
    frequency['Frequency'] = pd.cut(frequency['NumberOfOrders(F)'],
                                    bins = [-1,3,8,35,300],
                                    labels = [1,2,3,4]).astype(int)
    monetary['Monetary'] = pd.cut(monetary['TotalSpent(M)'],
                                    bins = [-1,1000,10000,100000,1000000],
                                    labels = [1,2,3,4]).astype(int)
    
    
    # Again, some plots to see the scorings (more uniform for recency, log for freq + monetary)
    recency.hist(column = 'Recency')
    frequency.hist(column = 'Frequency')
    monetary.hist(column = 'Monetary')
    
    # Recombine, create score
    customer_rfm = recency.merge(frequency, on = 'Customer ID', validate = 'one_to_one').merge(monetary, on = 'Customer ID', validate = 'one_to_one')
    customer_rfm['RFMScore'] = customer_rfm['Recency'].astype(str) + customer_rfm['Frequency'].astype(str) + customer_rfm['Monetary'].astype(str)
    
    # Create groupings of customers (a couple extreme cases, and some averages)
    
    customer_rfm.loc[(customer_rfm['Recency'] + customer_rfm['Frequency'] + customer_rfm['Monetary']).isin([4,5,6]), 'Segment'] = 'Bad Customer'
    customer_rfm.loc[(customer_rfm['Recency'] + customer_rfm['Frequency'] + customer_rfm['Monetary']).isin([7,8,9]), 'Segment'] = 'Average Customer'
    customer_rfm.loc[(customer_rfm['Recency'] + customer_rfm['Frequency'] + customer_rfm['Monetary']).isin([10,11,12,13]), 'Segment'] = 'Good Customer'
    
    customer_rfm.loc[customer_rfm['RFMScore'].str[0] == '1', 'Segment'] = 'Lost Customers'
    customer_rfm.loc[customer_rfm['RFMScore'].str[1] == '1', 'Segment'] = 'Infrequent Shopper'
    customer_rfm.loc[customer_rfm['RFMScore'].str[2] == '1', 'Segment'] = 'Small Spender'
    customer_rfm.loc[customer_rfm['RFMScore'] == '111', 'Segment'] = 'Worst Customers'
    
    customer_rfm.loc[customer_rfm['RFMScore'].str[0] == '4', 'Segment'] = 'Recent Customers'
    customer_rfm.loc[customer_rfm['RFMScore'].str[1] == '4', 'Segment'] = 'Frequent Shoppers'
    customer_rfm.loc[customer_rfm['RFMScore'].str[2] == '4', 'Segment'] = 'Big Spenders'
    customer_rfm.loc[customer_rfm['RFMScore'] == '444', 'Segment'] = 'Best Customers'
    
    return customer_rfm

def __product_tagging(products):
    """
    TODO
    """
    return None

def modelling_main(df_cleaned):
    """
    pd.DataFrame -> dict ( pd.DataFrame )
    
    Uses RFM identification and product tagging for the cleaned data to break
    out categories for later visualisation
    """
    # Create dict to write our dataframes for modelling to
    modelling_datasets = {}
    
    # Subset our data for RFM modelling and product tagging
    rfm_data, products = __subset_useful_data(df_cleaned)
    
    # Get customer_rfm scorings
    customer_rfm = __RFM_identification(rfm_data)
    
    # Get product tags 
    # TODO: This would be more involved, using sklearn and some NLP to categorise
    product_groups = pd.DataFrame()
    
    modelling_datasets['Customers'] = customer_rfm
    modelling_datasets['Products'] = product_groups
    
    return modelling_datasets

