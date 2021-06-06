import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


## acquire data

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the sql database.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


def get_telco_data():
    '''
    This function reads in the telco churn data from the sql database
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT *
    FROM payment_types
    JOIN customers ON payment_types.payment_type_id = customers.payment_type_id
    JOIN internet_service_types ON internet_service_types.internet_service_type_id = customers.internet_service_type_id
    JOIN contract_types ON contract_types.contract_type_id = customers.contract_type_id;
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))



def prep_telco(df):
    '''
    This function takes in the telco dataframe and specifies it to two year customers.
    It also replaces whitespaces with nulls, drops nulls, and changes total charges to
    a float. In the end it only returns specified columns.
    '''
    # isolate the customers on two year plans
    df = df[df['contract_type']=='Two year']

    # assign my dataframe to the columns I want
    column = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    df = df[column]

    # replace my empty whitespaces with nan values
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # drop my nulls
    df = df.dropna()

    # change my total charges column to a float
    df.total_charges = df.total_charges.astype('float')

    # replace 0 tenure months with 1
    df.tenure = df.tenure.replace(0, 1)

    # reset index
    df.reset_index(inplace=True)

    return df


def telco_split(df):
    '''
    This function takes in a dataframe and splits it into train, test, and validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test


## *********** Zillow data ****************

def get_zillow():
    '''
    This function reads in the zillow data from the sql database
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT bathroomcnt, 
			calculatedfinishedsquarefeet,
			taxvaluedollarcnt,
			yearbuilt,
			taxamount,
			fips
    FROM properties_2017
    WHERE propertylandusetypeid LIKE 261;
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))



def prep_zillow(df):
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df
